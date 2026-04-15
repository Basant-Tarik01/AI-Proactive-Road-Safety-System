import carla, traci
import os, json, math, random, argparse, threading, time
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--sumo-cfg",     required=True)
parser.add_argument("--steps",        type=int,   default=600)
parser.add_argument("--output-dir",   default="dataset")
parser.add_argument("--danger-ratio", type=float, default=0.35)
parser.add_argument("--seq-len",      type=int,   default=60)
parser.add_argument("--warmup",       type=int,   default=80)
parser.add_argument("--min-actors",   type=int,   default=3)
parser.add_argument("--enable-pedestrians", action="store_true", 
                    help="Enable pedestrian spawning from SUMO")
parser.add_argument("--max-pedestrians", type=int, default=8,
                    help="Maximum number of pedestrians to spawn (recommended: 5-10)")
args = parser.parse_args()

for d in ["rgb","depth","lidar","metadata","sequences"]:
    os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)

print(f"[INFO] Output: {args.output_dir}/  Steps: {args.steps}  "
      f"Danger: {int(args.danger_ratio*100)}%  SeqLen: {args.seq_len}")
print(f"[INFO] Pedestrians: {'ENABLED' if args.enable_pedestrians else 'DISABLED'} "
      f"(max: {args.max_pedestrians})")

# ─────────────────────────────────────────────────────────────────────────────
# CARLA connect with retry logic
# ─────────────────────────────────────────────────────────────────────────────
def connect_to_carla(max_retries=5):
    """Connect to CARLA with retry logic"""
    for attempt in range(max_retries):
        try:
            client = carla.Client("localhost", 2000)
            client.set_timeout(20.0)  # Increased timeout
            world = client.get_world()
            print(f"[INFO] Connected to CARLA (attempt {attempt+1}/{max_retries})")
            return client, world
        except Exception as e:
            print(f"[WARN] Connection attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise

client, world = connect_to_carla()
carla_map = world.get_map()

settings = world.get_settings()
settings.synchronous_mode    = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

blueprints = world.get_blueprint_library()

traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_global_distance_to_leading_vehicle(2.5)
traffic_manager.global_percentage_speed_difference(-20)

# Only keeps 4-wheeled cars (no trucks/buses).
BLOCKED = {"truck","bus","van","ambulance","firetruck","sprinter","carlacola","fusorosa"}
vehicle_bps = [
    bp for bp in blueprints.filter("vehicle.*")
    if int(bp.get_attribute("number_of_wheels")) == 4
    and not any(k in bp.id for k in BLOCKED)
]

# Pedestrian blueprints
pedestrian_bps = [bp for bp in blueprints.filter("walker.pedestrian.*")]
walker_controller_bp = blueprints.find('controller.ai.walker')
print(f"[INFO] Found {len(pedestrian_bps)} pedestrian blueprints")

# ─────────────────────────────────────────────────────────────────────────────
# SUMO + Ego
# ─────────────────────────────────────────────────────────────────────────────
traci.start(["sumo", "-c", args.sumo_cfg])

spawn_points = carla_map.get_spawn_points()
ego = world.spawn_actor(blueprints.find("vehicle.tesla.model3"), spawn_points[0])
ego.set_autopilot(True, traffic_manager.get_port())
traffic_manager.vehicle_percentage_speed_difference(ego, -30)
print("[INFO] Ego spawned.")

# ─────────────────────────────────────────────────────────────────────────────
# Sensors
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_T = carla.Transform(carla.Location(x=1.5, z=2.4))

def make_cam(name, w="800", h="600", fov="90"):
    bp = blueprints.find(name)
    bp.set_attribute("image_size_x", w)
    bp.set_attribute("image_size_y", h)
    bp.set_attribute("fov", fov)
    return world.spawn_actor(bp, SENSOR_T, attach_to=ego)

rgb_cam   = make_cam("sensor.camera.rgb")
depth_cam = make_cam("sensor.camera.depth")

lidar_bp = blueprints.find("sensor.lidar.ray_cast")
for k,v in [("channels","64"),("range","100"),("points_per_second","1120000"),
            ("rotation_frequency","20"),("upper_fov","10"),("lower_fov","-30")]:
    lidar_bp.set_attribute(k, v)
lidar_sensor = world.spawn_actor(lidar_bp,
    carla.Transform(carla.Location(x=0.0, z=2.8)), attach_to=ego)

rgb_ev=threading.Event(); depth_ev=threading.Event(); lidar_ev=threading.Event()
latest={"rgb":None,"depth":None,"lidar":None}
buf_lock=threading.Lock()

def _cb(key, ev):
    def _f(x):
        with buf_lock: latest[key]=x
        ev.set()
    return _f

rgb_cam.listen(_cb("rgb", rgb_ev))
depth_cam.listen(_cb("depth", depth_ev))
lidar_sensor.listen(_cb("lidar", lidar_ev))
spectator = world.get_spectator()

# ─────────────────────────────────────────────────────────────────────────────
# Decode helpers
# ─────────────────────────────────────────────────────────────────────────────
def decode_depth_mm(img):
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
    B,G,R = arr[:,:,0].astype(np.float32), arr[:,:,1].astype(np.float32), arr[:,:,2].astype(np.float32)
    return ((R + G*256.0 + B*65536.0) / (256.0**3-1) * 1e6).clip(0, 65535).astype(np.uint16)

def save_depth(mm, path):
    try:
        from PIL import Image
        Image.fromarray(mm, "I;16").save(path)
    except ImportError:
        np.save(path.replace(".png",".npy"), mm)

def decode_lidar(pc):
    return np.frombuffer(pc.raw_data, dtype=np.float32).reshape((-1, 4))

# ─────────────────────────────────────────────────────────────────────────────
# Vehicle slot grid + helpers
# ─────────────────────────────────────────────────────────────────────────────
SLOT_GRID = [
    (15,0),(25,0),(40,0),(55,0),
    (-8,0),(-15,0),(-25,0),(-40,0),
    (8,-3.5),(18,-3.5),(30,-3.5),(45,-3.5),(-8,-3.5),(-18,-3.5),(-30,-3.5),
    (8,3.5),(18,3.5),(30,3.5),(45,3.5),(-8,3.5),(-18,3.5),(-30,3.5),
    (10,-7.0),(22,-7.0),(35,-7.0),(-10,-7.0),(-22,-7.0),
    (10,7.0),(22,7.0),(35,7.0),(-10,7.0),(-22,7.0),
]

# IMPROVED: Pedestrian slot grid - further from road, safer positions
PED_SLOT_GRID = [
    # Far sidewalk left - safe distance from vehicles
    (15, -6.0), (25, -6.0), (35, -6.0), (45, -6.0),
    (20, -7.5), (30, -7.5), (40, -7.5),
    # Far sidewalk right - safe distance from vehicles
    (15, 6.0), (25, 6.0), (35, 6.0), (45, 6.0),
    (20, 7.5), (30, 7.5), (40, 7.5),
    # Sparse crossing positions - only for danger scenarios
    (18, -2.5), (18, 2.5), (28, -2.5), (28, 2.5),
]

actors = {}  # Vehicle actors
pedestrians = {}  # Pedestrian actors
ped_controllers = {}  
vehicle_slots = {}
ped_slots = {}
used_slots = set()
used_ped_slots = set()
npc_prev_loc = {}
ped_prev_loc = {}

# Track spawn failures
ped_spawn_failures = 0
max_ped_spawn_failures = 100  # Increased threshold

def get_wp(ego_loc, ego_yaw, fwd, lat):
    # Transform relative coordinates to world coordinates
    rad = math.radians(ego_yaw)
    x = ego_loc.x + fwd*math.cos(rad) - lat*math.sin(rad)
    y = ego_loc.y + fwd*math.sin(rad) + lat*math.cos(rad)
    # Project to nearest drivable road
    wp = carla_map.get_waypoint(carla.Location(x=x,y=y,z=ego_loc.z),
                                project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None: return None, None
    loc = wp.transform.location; loc.z += 0.3
    return loc, wp.transform.rotation.yaw

def get_sidewalk_loc(ego_loc, ego_yaw, fwd, lat):
    """Get location for pedestrian - tries sidewalk, then shoulder, then road"""
    rad = math.radians(ego_yaw)
    x = ego_loc.x + fwd*math.cos(rad) - lat*math.sin(rad)
    y = ego_loc.y + fwd*math.sin(rad) + lat*math.cos(rad)
    
    # Try sidewalk first
    wp = carla_map.get_waypoint(carla.Location(x=x,y=y,z=ego_loc.z),
                                project_to_road=True, lane_type=carla.LaneType.Sidewalk)
    if wp is not None:
        loc = wp.transform.location
        loc.z += 0.3  # Higher to avoid ground collision
        return loc, wp.transform.rotation.yaw
    
    # Try shoulder/parking
    wp = carla_map.get_waypoint(carla.Location(x=x,y=y,z=ego_loc.z),
                                project_to_road=True, lane_type=carla.LaneType.Shoulder)
    if wp is not None:
        loc = wp.transform.location
        loc.z += 0.3
        return loc, wp.transform.rotation.yaw
    
    # Last resort: try driving lane edge (for crosswalks)
    wp = carla_map.get_waypoint(carla.Location(x=x,y=y,z=ego_loc.z),
                                project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is not None:
        loc = wp.transform.location
        # Offset to edge of lane
        loc.x += lat * 0.3 * math.sin(rad)
        loc.y -= lat * 0.3 * math.cos(rad)
        loc.z += 0.3
        return loc, wp.transform.rotation.yaw
    
    return None, None

def assign_slot(vid):
    avail = [i for i in range(len(SLOT_GRID)) if i not in used_slots]
    if not avail: return None
    idx = random.choice(avail)
    vehicle_slots[vid]=idx; used_slots.add(idx); return idx

def assign_ped_slot(pid):
    avail = [i for i in range(len(PED_SLOT_GRID)) if i not in used_ped_slots]
    if not avail: return None
    idx = random.choice(avail)
    ped_slots[pid]=idx; used_ped_slots.add(idx); return idx

def release_slot(vid):
    if vid in vehicle_slots:
        used_slots.discard(vehicle_slots[vid]); del vehicle_slots[vid]

def release_ped_slot(pid):
    if pid in ped_slots:
        used_ped_slots.discard(ped_slots[pid]); del ped_slots[pid]

def blocking(loc, ego_loc, ego_yaw):
    rad = math.radians(ego_yaw)
    dx,dy = loc.x-ego_loc.x, loc.y-ego_loc.y
    return (0 < dx*math.cos(rad)+dy*math.sin(rad) < 12 and
            abs(-dx*math.sin(rad)+dy*math.cos(rad)) < 2.5)

def npc_velocity(vid, loc, step, dt=0.05):
    if vid in npc_prev_loc:
        px,py,ps = npc_prev_loc[vid]
        elapsed = max(1, step-ps) * dt
        vx = (loc.x-px)/elapsed; vy = (loc.y-py)/elapsed
        spd = math.sqrt(vx*vx+vy*vy)
    else:
        vx=vy=spd=0.0
    npc_prev_loc[vid] = (loc.x, loc.y, step)
    return vx, vy, spd

def ped_velocity(pid, loc, step, dt=0.05):
    if pid in ped_prev_loc:
        px,py,ps = ped_prev_loc[pid]
        elapsed = max(1, step-ps) * dt
        vx = (loc.x-px)/elapsed; vy = (loc.y-py)/elapsed
        spd = math.sqrt(vx*vx+vy*vy)
    else:
        vx=vy=spd=0.0
    ped_prev_loc[pid] = (loc.x, loc.y, step)
    return vx, vy, spd

# ─────────────────────────────────────────────────────────────────────────────
# Lane-width-aware risk scoring (includes pedestrians)
# ─────────────────────────────────────────────────────────────────────────────
SAME_LANE_LAT_THRESHOLD = 1.8
PED_RISK_MULTIPLIER = 1.5  # Pedestrians are higher priority in risk calculation

def compute_risk(ego_loc, ego_yaw, ego_vx, ego_vy, actors_dict, peds_dict, step, is_danger_scenario):
    max_risk = 0.0
    ttc_min  = 999.0
    min_dist = 999.0
    details  = []
    rad      = math.radians(ego_yaw)
    cos_y    = math.cos(rad)
    sin_y    = math.sin(rad)

    # Process vehicles
    for vid, actor in actors_dict.items():
        try:
            npc_loc = actor.get_location()
            npc_yaw = actor.get_transform().rotation.yaw
        except Exception:
            continue

        dx   = npc_loc.x - ego_loc.x
        dy   = npc_loc.y - ego_loc.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.5: continue
        min_dist = min(min_dist, dist)

        fwd_dist = dx*cos_y + dy*sin_y
        lat_dist = abs(-dx*sin_y + dy*cos_y)

        npc_vx, npc_vy, npc_spd = npc_velocity(vid, npc_loc, step)

        rel_vx = npc_vx - ego_vx
        rel_vy = npc_vy - ego_vy
        closing = -(dx*rel_vx + dy*rel_vy) / max(dist, 0.1)

        ttc = dist / closing if closing > 0.5 else 999.0
        ttc_min = min(ttc_min, ttc)

        risk = 0.0

        if ttc < 4.0 and closing > 0.5:
            r_time  = (4.0 - ttc) / 4.0
            r_close = min(1.0, closing / 15.0)
            r_near  = min(1.0, max(0.0, 30.0 - dist) / 30.0)
            risk_ttc = r_time*0.5 + r_close*0.3 + r_near*0.2
            risk = max(risk, risk_ttc)

        if fwd_dist > 0 and lat_dist <= SAME_LANE_LAT_THRESHOLD and fwd_dist < 25.0:
            r_prox = max(0.0, (25.0 - fwd_dist) / 25.0)
            npc_cos = math.cos(math.radians(npc_yaw))
            npc_sin = math.sin(math.radians(npc_yaw))
            dot = cos_y*npc_cos + sin_y*npc_sin
            if dot < -0.5:
                r_prox = min(1.0, r_prox * 1.5)
            risk = max(risk, r_prox * 0.85)

        if not is_danger_scenario:
            risk = min(risk, 0.40)

        max_risk = max(max_risk, risk)

        if risk > 0.05:
            details.append({
                "actor_id":     vid,
                "actor_type":   "vehicle",
                "dist_m":       round(dist,    2),
                "fwd_m":        round(fwd_dist,2),
                "lat_m":        round(lat_dist,2),
                "ttc_s":        round(ttc,     2),
                "closing_ms":   round(closing, 2),
                "speed_ms":     round(npc_spd, 2),
                "risk":         round(risk,    4),
            })

    # Process pedestrians with higher risk weighting
    for pid, ped_actor in peds_dict.items():
        try:
            ped_loc = ped_actor.get_location()
        except Exception:
            continue

        dx   = ped_loc.x - ego_loc.x
        dy   = ped_loc.y - ego_loc.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.5: continue
        min_dist = min(min_dist, dist)

        fwd_dist = dx*cos_y + dy*sin_y
        lat_dist = abs(-dx*sin_y + dy*cos_y)

        ped_vx, ped_vy, ped_spd = ped_velocity(pid, ped_loc, step)

        rel_vx = ped_vx - ego_vx
        rel_vy = ped_vy - ego_vy
        closing = -(dx*rel_vx + dy*rel_vy) / max(dist, 0.1)

        # Pedestrians in front of vehicle = very high risk
        risk = 0.0
        if fwd_dist > 0 and fwd_dist < 20.0 and lat_dist < 4.0:
            # Proximity risk for pedestrians
            r_prox = max(0.0, (20.0 - fwd_dist) / 20.0)
            
            # Higher risk if pedestrian is crossing (lateral position changing)
            if abs(ped_vx) > 0.1 or abs(ped_vy) > 0.1:
                r_prox = min(1.0, r_prox * 1.3)
            
            risk = r_prox * 0.9 * PED_RISK_MULTIPLIER
        
        # TTC-based risk for moving pedestrians
        if closing > 0.3:
            ttc = dist / closing
            ttc_min = min(ttc_min, ttc)
            if ttc < 3.0:
                r_time = (3.0 - ttc) / 3.0
                risk = max(risk, r_time * 0.85 * PED_RISK_MULTIPLIER)

        if not is_danger_scenario:
            risk = min(risk, 0.50)  # Allow slightly higher base risk for pedestrians

        max_risk = max(max_risk, risk)

        if risk > 0.05:
            details.append({
                "actor_id":     pid,
                "actor_type":   "pedestrian",
                "dist_m":       round(dist,    2),
                "fwd_m":        round(fwd_dist,2),
                "lat_m":        round(lat_dist,2),
                "ttc_s":        round(ttc if closing > 0.3 else 999, 2),
                "closing_ms":   round(closing, 2),
                "speed_ms":     round(ped_spd, 2),
                "risk":         round(risk,    4),
            })

    return (round(max_risk,4), round(ttc_min,2), round(min_dist,2),
            sorted(details, key=lambda x: -x["risk"])[:8])  # Show top 8 threats


def risk_level(s):
    if s > 0.70: return "critical"
    if s > 0.40: return "high"
    if s > 0.15: return "medium"
    return "safe"

# ─────────────────────────────────────────────────────────────────────────────
# Danger scenario injector (enhanced with pedestrian scenarios)
# ─────────────────────────────────────────────────────────────────────────────
SCENARIO_TYPES = ["cut_in","sudden_brake","head_on_close","ped_crossing","ped_jaywalking"]

def inject_danger(ego_loc, ego_yaw):
    scenario = random.choice(SCENARIO_TYPES)
    rad = math.radians(ego_yaw)
    
    # Pedestrian scenarios
    if scenario in ["ped_crossing", "ped_jaywalking"] and pedestrians:
        pid = random.choice(list(pedestrians.keys()))
        
        if scenario == "ped_crossing":
            # Pedestrian crossing in front of vehicle
            fwd = random.uniform(8, 15)
            lat = random.uniform(-2.0, 2.0)
        else:  # ped_jaywalking
            # Pedestrian suddenly appearing from side
            fwd = random.uniform(5, 12)
            lat = random.choice([-4.0, 4.0])
        
        loc = carla.Location(
            x = ego_loc.x + fwd*math.cos(rad) - lat*math.sin(rad),
            y = ego_loc.y + fwd*math.sin(rad) + lat*math.cos(rad),
            z = ego_loc.z + 0.3
        )
        
        try:
            # Teleport pedestrian
            pedestrians[pid].set_location(loc)
            
            # Update AI controller target to cross road
            if pid in ped_controllers:
                # Make pedestrian walk toward opposite side
                crossing_target = carla.Location(
                    x = loc.x - lat*2*math.sin(rad),
                    y = loc.y + lat*2*math.cos(rad),
                    z = loc.z
                )
                ped_controllers[pid].go_to_location(crossing_target)
                ped_controllers[pid].set_max_speed(1.8)  # Faster for jaywalking
            
            ped_prev_loc.pop(pid, None)
            return scenario
        except Exception:
            return "normal"
    
    # Vehicle scenarios
    elif actors:
        vid = random.choice(list(actors.keys()))

        if scenario == "cut_in":
            fwd = random.uniform(10,16); lat = random.choice([-3.5,3.5])
        elif scenario == "sudden_brake":
            fwd = random.uniform(10,18); lat = random.uniform(-1.0,1.0)
        else:  # head_on_close
            fwd = random.uniform(12,25); lat = random.uniform(-0.5,0.5)

        loc = carla.Location(
            x = ego_loc.x + fwd*math.cos(rad) - lat*math.sin(rad),
            y = ego_loc.y + fwd*math.sin(rad) + lat*math.cos(rad),
            z = ego_loc.z + 0.3
        )
        yaw = ego_yaw if scenario != "head_on_close" else ego_yaw + 180

        wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp:
            loc = wp.transform.location; loc.z += 0.3
            if scenario != "head_on_close": yaw = wp.transform.rotation.yaw

        try:
            actors[vid].set_transform(carla.Transform(loc, carla.Rotation(yaw=yaw)))
            npc_prev_loc.pop(vid, None)
            return scenario
        except Exception:
            return "normal"
    
    return "normal"

# ─────────────────────────────────────────────────────────────────────────────
# Sequence tracker
# ─────────────────────────────────────────────────────────────────────────────
class SeqTracker:
    def __init__(self, n): self.n=n; self.buf=[]; self.seqs=[]
    def push(self, fid, risk, scen):
        self.buf.append({"frame":fid,"risk":risk,"scenario":scen})
        if len(self.buf) >= self.n: self._flush()
    def _flush(self):
        w=self.buf[:self.n]; self.buf=self.buf[self.n//2:]
        mr=max(f["risk"] for f in w)
        sc=list({f["scenario"] for f in w}-{"normal"})
        self.seqs.append({
            "seq_id":     len(self.seqs),
            "frames":     [f["frame"] for f in w],
            "risk_label": round(mr,4),
            "risk_level": risk_level(mr),
            "scenarios":  sc or ["normal"],
            "frame_risks":[round(f["risk"],4) for f in w],
        })
    def flush_remaining(self):
        while len(self.buf) >= max(10, self.n//4):
            w=self.buf; self.buf=[]
            mr=max(f["risk"] for f in w)
            sc=list({f["scenario"] for f in w}-{"normal"})
            self.seqs.append({
                "seq_id":     len(self.seqs),
                "frames":     [f["frame"] for f in w],
                "risk_label": round(mr,4),
                "risk_level": risk_level(mr),
                "scenarios":  sc or ["normal"],
                "frame_risks":[round(f["risk"],4) for f in w],
                "partial":    True,
            })
            break
    def save(self, path):
        self.flush_remaining()
        with open(path,"w") as f:
            json.dump(self.seqs, f, indent=2)
        return len(self.seqs)

seq_tracker = SeqTracker(args.seq_len)
stats = {"saved":0,"skipped":0,"danger":0,"timeout":0,
         "risk":{"safe":0,"medium":0,"high":0,"critical":0},
         "scen":{"normal":0,"cut_in":0,"sudden_brake":0,"head_on_close":0,
                 "ped_crossing":0,"ped_jaywalking":0},
         "ped_spawn_attempts":0,"ped_spawn_success":0}


simulation_complete = False
current_scenario    = "normal"
step                = 0
consecutive_failures = 0  # Track sensor failures
max_consecutive_failures = 50  # More lenient threshold
last_successful_tick = 0

def safe_tick():
    """Safely tick the world with retry logic"""
    global consecutive_failures, last_successful_tick
    max_retries = 3
    for attempt in range(max_retries):
        try:
            world.tick()
            consecutive_failures = 0
            last_successful_tick = step
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WARN] Tick failed (attempt {attempt+1}), retrying...")
                time.sleep(0.1)
            else:
                consecutive_failures += 1
                print(f"[ERROR] Tick failed after {max_retries} attempts: {e}")
                return False
    return False

try:
    print("[INFO] Starting simulation ...")

    for step in range(args.steps):
        traci.simulationStep()
        rgb_ev.clear(); depth_ev.clear(); lidar_ev.clear()
        
        # Robust tick with recovery
        if not safe_tick():
            if consecutive_failures >= max_consecutive_failures:
                print(f"[ERROR] Too many consecutive failures ({consecutive_failures}). Stopping.")
                break
            continue

        # Generous sensor timeouts with graceful degradation
        got_rgb   = rgb_ev.wait(timeout=5.0)
        got_depth = depth_ev.wait(timeout=5.0)
        got_lidar = lidar_ev.wait(timeout=5.0)

        if not (got_rgb and got_depth and got_lidar):
            stats["timeout"] += 1
            stats["skipped"] += 1
            consecutive_failures += 1
            
            # Log every 15 timeouts
            if stats["timeout"] % 15 == 1:
                print(f"[WARN] Step {step}: sensor timeout #{stats['timeout']} "
                      f"(cars={len(actors)}, peds={len(pedestrians)})")
            
            # Only abort on extreme consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                print(f"[ERROR] {consecutive_failures} consecutive sensor failures. Aborting.")
                break
            
            continue

        # Reset consecutive failures on success
        consecutive_failures = 0
        
        sumo_veh_ids = traci.vehicle.getIDList()
        sumo_ped_ids = traci.person.getIDList() if args.enable_pedestrians else []
        
        ego_t    = ego.get_transform()
        ego_loc  = ego_t.location
        ego_yaw  = ego_t.rotation.yaw
        ego_vel  = ego.get_velocity()
        ego_vx, ego_vy = ego_vel.x, ego_vel.y
        ego_spd  = math.sqrt(ego_vx**2 + ego_vy**2) * 3.6

        spectator.set_transform(carla.Transform(
            ego_loc + carla.Location(z=40), carla.Rotation(pitch=-90)
        ))

        if step < args.warmup:
            if step % 10 == 0:
                print(f"[WARMUP {step}/{args.warmup}] ego={ego_spd:.1f}km/h "
                      f"vehs={len(sumo_veh_ids)} peds={len(sumo_ped_ids)}")
            continue

        if ego_spd < 1.0 and step < args.warmup + 50:
            print(f"[WAIT] Step {step}: ego stationary ({ego_spd:.1f} km/h)")
            stats["skipped"] += 1
            continue

        # Danger injection
        inject_this = random.random() < args.danger_ratio
        if inject_this and (actors or pedestrians):
            current_scenario = inject_danger(ego_loc, ego_yaw)
            stats["danger"] += 1
        else:
            current_scenario = "normal"

        is_danger = current_scenario != "normal"

        # ═══════════════════════════════════════════════════════════════════
        # VEHICLE SPAWN/UPDATE
        # ═══════════════════════════════════════════════════════════════════
        for vid in sumo_veh_ids:
            if vid not in actors:
                idx = assign_slot(vid)
                if idx is None: continue
                fwd,lat = SLOT_GRID[idx]
                loc,yaw = get_wp(ego_loc, ego_yaw, fwd, lat)
                if loc is None or blocking(loc,ego_loc,ego_yaw):
                    release_slot(vid); continue
                bp = random.choice(vehicle_bps)
                try:
                    a = world.spawn_actor(bp, carla.Transform(loc,carla.Rotation(yaw=yaw)))
                    a.set_simulate_physics(False)
                    actors[vid] = a
                except Exception:
                    release_slot(vid)
            else:
                if inject_this and current_scenario not in ["ped_crossing","ped_jaywalking"]:
                    continue
                idx = vehicle_slots.get(vid)
                if idx is None: continue
                fwd,lat = SLOT_GRID[idx]
                loc,yaw = get_wp(ego_loc, ego_yaw, fwd, lat)
                if loc is None: continue
                if blocking(loc,ego_loc,ego_yaw):
                    try: actors[vid].destroy()
                    except Exception: pass
                    del actors[vid]; release_slot(vid); continue
                try: actors[vid].set_transform(carla.Transform(loc,carla.Rotation(yaw=yaw)))
                except Exception: pass

        for vid in list(actors.keys()):
            if vid not in sumo_veh_ids:
                try: actors[vid].destroy()
                except Exception: pass
                del actors[vid]; release_slot(vid); npc_prev_loc.pop(vid,None)

        # ═══════════════════════════════════════════════════════════════════
        # PEDESTRIAN SPAWN/UPDATE - FIXED WITH PROPER AI CONTROLLER INIT
        # ═══════════════════════════════════════════════════════════════════
        if args.enable_pedestrians and ped_spawn_failures < max_ped_spawn_failures:
            current_ped_count = len(pedestrians)
            can_spawn_new = current_ped_count < args.max_pedestrians
            
            # Throttle spawning - max 1 new pedestrian per step
            spawned_this_step = 0
            max_spawns_per_step = 1
            
            if can_spawn_new and len(sumo_ped_ids) > 0:
                for pid in sumo_ped_ids:
                    if pid not in pedestrians and spawned_this_step < max_spawns_per_step:
                        stats["ped_spawn_attempts"] += 1
                        idx = assign_ped_slot(pid)
                        if idx is None: 
                            continue
                        
                        fwd, lat = PED_SLOT_GRID[idx]
                        loc, yaw = get_sidewalk_loc(ego_loc, ego_yaw, fwd, lat)
                        if loc is None: 
                            release_ped_slot(pid)
                            ped_spawn_failures += 1
                            continue
                        
                        # Check distance from ego to avoid immediate collisions
                        dist_to_ego = math.sqrt((loc.x-ego_loc.x)**2 + (loc.y-ego_loc.y)**2)
                        if dist_to_ego < 5.0:
                            release_ped_slot(pid)
                            continue
                        
                        bp = random.choice(pedestrian_bps)
                        try:
                            # CRITICAL FIX: Proper initialization sequence for standing pedestrians
                            
                            # Step 1: Spawn pedestrian actor
                            ped_actor = world.spawn_actor(
                                bp, 
                                carla.Transform(loc, carla.Rotation(yaw=yaw))
                            )
                            
                            # Step 2: Let physics settle (ensures proper pose initialization)
                            safe_tick()
                            time.sleep(0.02)  # Small delay for physics
                            
                            # Step 3: Spawn AI controller and attach
                            controller = world.spawn_actor(
                                walker_controller_bp,
                                carla.Transform(),
                                attach_to=ped_actor
                            )
                            
                            # Step 4: Let controller attach to actor
                            safe_tick()
                            
                            # Step 5: Start AI controller (CRITICAL for standing pose)
                            controller.start()
                            
                            # Step 6: Let AI initialize (transition from crouched to standing)
                            safe_tick()
                            time.sleep(0.02)
                            
                            # Step 7: Set walking behavior with varied targets
                            target_loc = carla.Location(
                                x = loc.x + random.uniform(-20, 20),
                                y = loc.y + random.uniform(-20, 20),
                                z = loc.z
                            )
                            controller.go_to_location(target_loc)
                            controller.set_max_speed(1.1 + random.uniform(-0.2, 0.4))
                            
                            # Step 8: Final tick to apply all changes
                            safe_tick()
                            
                            # Only add to dicts after complete initialization
                            pedestrians[pid] = ped_actor
                            ped_controllers[pid] = controller
                            spawned_this_step += 1
                            stats["ped_spawn_success"] += 1
                            
                        except Exception as e:
                            ped_spawn_failures += 1
                            if ped_spawn_failures % 25 == 1:
                                print(f"[WARN] Ped spawn failed #{ped_spawn_failures}: {e}")
                            release_ped_slot(pid)
                        
                        if len(pedestrians) >= args.max_pedestrians:
                            break
            
            # Update pedestrian positions (every 15 frames to reduce load)
            if step % 15 == 0:
                for pid in list(pedestrians.keys()):
                    if pid not in sumo_ped_ids:
                        # Clean up pedestrian and controller
                        try: 
                            if pid in ped_controllers:
                                ped_controllers[pid].stop()
                                ped_controllers[pid].destroy()
                                del ped_controllers[pid]
                        except Exception: pass
                        
                        try: 
                            pedestrians[pid].destroy()
                        except Exception: pass
                        
                        del pedestrians[pid]
                        release_ped_slot(pid)
                        ped_prev_loc.pop(pid, None)

        # Skip frames with too few total actors
        total_actors = len(actors) + len(pedestrians)
        if total_actors < args.min_actors:
            stats["skipped"] += 1
            continue

        with buf_lock:
            img_rgb=latest["rgb"]; img_depth=latest["depth"]; pc_lidar=latest["lidar"]

        frame = img_rgb.frame

        # Risk computation (includes pedestrians)
        risk_score, ttc_min, min_dist, risk_details = compute_risk(
            ego_loc, ego_yaw, ego_vx, ego_vy, actors, pedestrians, step, is_danger
        )
        rl = risk_level(risk_score)

        # Save RGB
        img_rgb.save_to_disk(os.path.join(args.output_dir,"rgb",f"{frame:06d}.png"))

        # Save depth
        save_depth(decode_depth_mm(img_depth),
                   os.path.join(args.output_dir,"depth",f"{frame:06d}.png"))

        # Save LiDAR
        points = decode_lidar(pc_lidar)
        np.save(os.path.join(args.output_dir,"lidar",f"{frame:06d}.npy"), points)

        # Camera intrinsics
        fov_r = math.radians(90)
        fx = (800/2.0)/math.tan(fov_r/2); fy = (600/2.0)/math.tan(fov_r/2)

        # NPC vehicle list
        npc_list = []
        for vid, actor in actors.items():
            try:
                t = actor.get_transform()
                vx,vy,spd = npc_velocity(vid, t.location, step)
                npc_list.append({
                    "id":vid,"type":"vehicle","blueprint":actor.type_id,
                    "loc":{"x":round(t.location.x,3),"y":round(t.location.y,3),"z":round(t.location.z,3)},
                    "yaw":round(t.rotation.yaw,2),
                    "velocity":{"x":round(vx,3),"y":round(vy,3)},
                    "speed_ms":round(spd,3),
                })
            except Exception: continue

        # Pedestrian list
        ped_list = []
        for pid, ped_actor in pedestrians.items():
            try:
                t = ped_actor.get_transform()
                vx,vy,spd = ped_velocity(pid, t.location, step)
                ped_list.append({
                    "id":pid,"type":"pedestrian","blueprint":ped_actor.type_id,
                    "loc":{"x":round(t.location.x,3),"y":round(t.location.y,3),"z":round(t.location.z,3)},
                    "yaw":round(t.rotation.yaw,2),
                    "velocity":{"x":round(vx,3),"y":round(vy,3)},
                    "speed_ms":round(spd,3),
                })
            except Exception: continue

        # Metadata
        metadata = {
            "frame":frame,"step":step,
            "ego":{
                "location":{"x":round(ego_loc.x,3),"y":round(ego_loc.y,3),"z":round(ego_loc.z,3)},
                "rotation":{"pitch":round(ego_t.rotation.pitch,3),
                            "yaw":round(ego_yaw,3),
                            "roll":round(ego_t.rotation.roll,3)},
                "velocity":{"x":round(ego_vx,3),"y":round(ego_vy,3)},
                "speed_kmh":round(ego_spd,2),
            },
            "risk":{
                "score":risk_score,
                "level":rl,
                "ttc_min_s":ttc_min,
                "min_dist_m":min_dist,
                "top_threats":risk_details,
            },
            "scenario":current_scenario,
            "vehicles":npc_list,
            "pedestrians":ped_list,
            "total_actors": total_actors,
            "sensors":{
                "rgb":{"file":f"rgb/{frame:06d}.png","width":800,"height":600,"fov":90},
                "depth":{"file":f"depth/{frame:06d}.png","unit":"mm_uint16",
                         "fx":round(fx,2),"fy":round(fy,2),"cx":400,"cy":300},
                "lidar":{"file":f"lidar/{frame:06d}.npy",
                         "columns":["x","y","z","intensity"],
                         "num_points":int(points.shape[0])},
            },
            "yolo_detections":[],
        }

        with open(os.path.join(args.output_dir,"metadata",f"{frame:06d}.json"),"w") as f:
            json.dump(metadata, f, indent=2)

        seq_tracker.push(frame, risk_score, current_scenario)
        stats["saved"]+=1; stats["risk"][rl]+=1
        stats["scen"][current_scenario] = stats["scen"].get(current_scenario,0)+1

        if step % 20 == 0:
            print(f"[STEP {step:4d}] frame={frame}  cars={len(actors)}  peds={len(pedestrians)}  "
                  f"risk={risk_score:.3f}({rl})  scen={current_scenario}  "
                  f"spd={ego_spd:.1f}km/h  ttc={ttc_min:.1f}s  lidar={points.shape[0]:,}")


    simulation_complete = True
    print("\n[INFO] Simulation loop complete - proceeding to cleanup.")

except KeyboardInterrupt:
    print(f"\n[WARN] Simulation interrupted by user at step {step}")
except Exception as e:
    print(f"\n[ERROR] Simulation crashed at step {step}: {e}")
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# ROBUST CLEANUP - Always executes, saves data even on crashes
# ─────────────────────────────────────────────────────────────────────────────
finally:
    print("\n[CLEANUP] Starting cleanup sequence...")

    # ── PRIORITY 1: Save sequences IMMEDIATELY (most important) ──────────────
    seq_path = os.path.join(args.output_dir, "sequences", "index.json")
    try:
        n_seq = seq_tracker.save(seq_path)
        print(f"[✓] Sequences saved: {n_seq} sequences → {seq_path}")
    except Exception as e:
        print(f"[✗] Failed to save sequences: {e}")
        n_seq = 0

    # ── PRIORITY 2: Stop sensors (prevent callbacks during cleanup) ──────────
    print(f"[CLEANUP] Stopping sensors...")
    for sensor_obj in [rgb_cam, depth_cam, lidar_sensor]:
        try: sensor_obj.stop()
        except Exception: pass

    # ── PRIORITY 3: Destroy pedestrian controllers (BEFORE actors) ───────────
    print(f"[CLEANUP] Destroying {len(ped_controllers)} pedestrian controllers...")
    for pid, controller in list(ped_controllers.items()):
        try: 
            controller.stop()
            time.sleep(0.01)  # Small delay for graceful stop
            controller.destroy()
        except Exception as e:
            pass
    ped_controllers.clear()

    # ── PRIORITY 4: Destroy pedestrian actors ─────────────────────────────────
    print(f"[CLEANUP] Destroying {len(pedestrians)} pedestrians...")
    for pid, ped in list(pedestrians.items()):
        try: 
            ped.destroy()
        except Exception: 
            pass
    pedestrians.clear()

    # ── PRIORITY 5: Destroy sensors ───────────────────────────────────────────
    print(f"[CLEANUP] Destroying sensors...")
    for sensor_obj in [rgb_cam, depth_cam, lidar_sensor]:
        try: sensor_obj.destroy()
        except Exception: pass

    # ── PRIORITY 6: Destroy NPC vehicles ──────────────────────────────────────
    print(f"[CLEANUP] Destroying {len(actors)} vehicles...")
    for vid, a in list(actors.items()):
        try: a.destroy()
        except Exception: pass
    actors.clear()

    # ── PRIORITY 7: Destroy ego vehicle ───────────────────────────────────────
    try: 
        ego.destroy()
        print("[✓] Ego vehicle destroyed")
    except Exception as e:
        print(f"[✗] Ego destruction failed: {e}")

    # ── PRIORITY 8: Close SUMO ────────────────────────────────────────────────
    try: 
        traci.close()
        print("[✓] SUMO closed")
    except Exception as e:
        print(f"[✗] SUMO close failed: {e}")

    # ── PRIORITY 9: Restore CARLA async mode ──────────────────────────────────
    try:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("[✓] CARLA async mode restored")
    except Exception as e:
        print(f"[✗] CARLA settings restore failed: {e}")

    # ── FINAL STATS ───────────────────────────────────────────────────────────
    total = stats["saved"]
    danger_frames = sum(stats["scen"].get(s,0) for s in 
                       ["cut_in","sudden_brake","head_on_close","ped_crossing","ped_jaywalking"])
    danger_pct = round(100*danger_frames/total,1) if total > 0 else 0
    ped_success_rate = round(100*stats["ped_spawn_success"]/max(stats["ped_spawn_attempts"],1),1)

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                     SIMULATION COMPLETE                          ║
╠══════════════════════════════════════════════════════════════════╣
║ Status           : {'✓ COMPLETE' if simulation_complete else '⚠ PARTIAL (crashed/interrupted)'}
║ Frames saved     : {stats['saved']}
║ Frames skipped   : {stats['skipped']}
║   ↳ sensor timeouts   : {stats['timeout']}
║   ↳ other reasons     : {stats['skipped'] - stats['timeout']}
║ Danger injections: {stats['danger']}
║ Sequences        : {n_seq}
╠══════════════════════════════════════════════════════════════════╣
║ PEDESTRIAN SPAWNING
║   Attempts       : {stats['ped_spawn_attempts']}
║   Successful     : {stats['ped_spawn_success']}
║   Success rate   : {ped_success_rate}%
╠══════════════════════════════════════════════════════════════════╣
║ RISK DISTRIBUTION
║   Safe           : {stats['risk']['safe']:4d}  ({100*stats['risk']['safe']//max(total,1):2d}%)
║   Medium         : {stats['risk']['medium']:4d}  ({100*stats['risk']['medium']//max(total,1):2d}%)
║   High           : {stats['risk']['high']:4d}  ({100*stats['risk']['high']//max(total,1):2d}%)
║   Critical       : {stats['risk']['critical']:4d}  ({100*stats['risk']['critical']//max(total,1):2d}%)
╠══════════════════════════════════════════════════════════════════╣
║ SCENARIO DISTRIBUTION
║   Normal         : {stats['scen'].get('normal',0):4d}
║   Cut-in         : {stats['scen'].get('cut_in',0):4d}
║   Sudden brake   : {stats['scen'].get('sudden_brake',0):4d}
║   Head-on        : {stats['scen'].get('head_on_close',0):4d}
║   Ped crossing   : {stats['scen'].get('ped_crossing',0):4d}
║   Ped jaywalking : {stats['scen'].get('ped_jaywalking',0):4d}
║
║   Danger ratio   : {danger_pct}% (target: 30-40%)
╠══════════════════════════════════════════════════════════════════╣
║ OUTPUT FILES
║   {args.output_dir}/rgb/          → {stats['saved']} frames
║   {args.output_dir}/depth/        → {stats['saved']} frames
║   {args.output_dir}/lidar/        → {stats['saved']} frames
║   {args.output_dir}/metadata/     → {stats['saved']} files
║   {args.output_dir}/sequences/    → {n_seq} sequences
╚══════════════════════════════════════════════════════════════════╝
""")