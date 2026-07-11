import os
import json
import math
import numpy as np
import pandas as pd
import cv2
import xgboost as xgb
from ultralytics import YOLO
from typing import Dict, List, Union, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# PerceptionPipeline  –  YOLO detection + feature extraction
# (extracted from the training notebook unchanged)
# ──────────────────────────────────────────────────────────────────────────────

class PerceptionPipeline:
    """
    YOLO-based perception pipeline to consume CARLA/SUMO ground-truth
    metadata alongside RGB detections.
    """

    def __init__(self,
                 model_path: str = 'yolov8n.pt',
                 conf_threshold: float = 0.4,
                 camera_config: Optional[Dict] = None):

        self.model           = YOLO(model_path)
        self.conf_threshold  = conf_threshold
        self.camera_config   = camera_config or self._default_camera_config()

        self.yolo_to_actor_type = {
            'car':           'vehicle',
            'truck':         'vehicle',
            'bus':           'vehicle',
            'person':        'pedestrian',
            'bicycle':       'cyclist',
            'motorcycle':    'cyclist',
            'traffic light': 'traffic_light',
            'stop sign':     'traffic_sign',
        }
        print(f"[PerceptionPipeline] model={model_path}  conf≥{conf_threshold}")

    def _default_camera_config(self) -> Dict:
        fov_r = math.radians(90)
        w, h  = 800, 600
        fx    = (w / 2.0) / math.tan(fov_r / 2)
        return {
            'focal_length':  fx,
            'image_width':   w,
            'image_height':  h,
            'camera_height': 2.4,
            'pitch_angle':   0.0,
            'x_scale':       40.0,
            'y_scale_min':   5.0,
            'y_scale_max':   50.0,
        }

    def estimate_3d_position(self, bbox: Dict, img_shape: Tuple[int, int]) -> Dict:
        x_norm      = bbox['x_center']
        y_norm      = bbox['y_center']
        width_norm  = bbox['width']

        x_world = (x_norm - 0.5) * self.camera_config['x_scale']
        y_world = (self.camera_config['y_scale_min'] +
                   (1.0 - y_norm) * self.camera_config['y_scale_max'])
        z_world = 0.0

        return {
            'x':        float(x_world),
            'y':        float(y_world),
            'z':        float(z_world),
            'yaw':      0.0,
            'speed_ms': 0.0,
            'vx':       0.0,
            'vy':       0.0,
        }

    @staticmethod
    def _match_gt_actor(yolo_x: float, yolo_y: float,
                        gt_actors: List[Dict],
                        threshold_m: float = 8.0) -> Optional[Dict]:
        best, best_d = None, threshold_m
        for actor in gt_actors:
            loc  = actor.get('loc', {})
            dx   = loc.get('x', 0) - yolo_x
            dy   = loc.get('y', 0) - yolo_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_d:
                best, best_d = actor, dist
        return best

    @staticmethod
    def load_metadata(meta_path: str) -> Optional[Dict]:
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load metadata {meta_path}: {e}")
            return None

    def process_frame(self,
                      frame:         Union[str, np.ndarray],
                      frame_id:      int,
                      run_id:        str  = "perception_run",
                      metadata:      Optional[Dict] = None,
                      target_label:  Optional[int]  = None
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        # ── 0. Ego ground truth ─────────────────────────────────────────────
        if metadata:
            ego_info  = metadata.get('ego', {})
            ego_loc   = ego_info.get('location', {'x': 0, 'y': 0, 'z': 0})
            ego_spd   = ego_info.get('speed_kmh', 0.0) / 3.6
            ego_vel   = ego_info.get('velocity',  {'x': 0.0, 'y': 0.0})
            ego_vx    = ego_vel.get('x', 0.0)
            ego_vy    = ego_vel.get('y', 0.0)
            ego_yaw   = ego_info.get('rotation', {}).get('yaw', 0.0)

            gt_risk       = metadata.get('risk', {})
            gt_vehicles   = metadata.get('vehicles',    [])
            gt_peds       = metadata.get('pedestrians', [])
            gt_all_actors = gt_vehicles + gt_peds
            scenario      = metadata.get('scenario', 'normal')

            if target_label is None:
                level_map    = {'safe': 0, 'medium': 1, 'high': 2, 'critical': 3}
                target_label = level_map.get(gt_risk.get('level', 'safe'), 0)
        else:
            ego_spd = ego_vx = ego_vy = 0.0
            ego_yaw = 0.0
            ego_loc = {'x': 0, 'y': 0, 'z': 0}
            gt_risk = {}
            gt_all_actors = []
            scenario = 'normal'
            if target_label is None:
                target_label = 0

        # ── 1. YOLO detection ───────────────────────────────────────────────
        results   = self.model.predict(source=frame,
                                       conf=self.conf_threshold,
                                       verbose=False)[0]
        boxes     = results.boxes
        img_shape = results.orig_shape

        # ── 2. Build nodes ──────────────────────────────────────────────────
        nodes_data = []
        for i in range(len(boxes)):
            class_id   = int(boxes.cls[i])
            class_name = results.names[class_id]
            confidence = float(boxes.conf[i])
            actor_type = self.yolo_to_actor_type.get(class_name, 'unknown')

            bbox = {
                'x_center': float(boxes.xywhn[i][0]),
                'y_center': float(boxes.xywhn[i][1]),
                'width':    float(boxes.xywhn[i][2]),
                'height':   float(boxes.xywhn[i][3]),
            }

            est_pos = self.estimate_3d_position(bbox, img_shape)

            approx_world_x = ego_loc['x'] + est_pos['y'] * math.cos(math.radians(ego_yaw)) \
                           - est_pos['x'] * math.sin(math.radians(ego_yaw))
            approx_world_y = ego_loc['y'] + est_pos['y'] * math.sin(math.radians(ego_yaw)) \
                           + est_pos['x'] * math.cos(math.radians(ego_yaw))

            gt_match = self._match_gt_actor(approx_world_x, approx_world_y,
                                             gt_all_actors) if gt_all_actors else None

            if gt_match:
                gt_loc = gt_match.get('loc', {})
                rad  = math.radians(ego_yaw)
                dx   = gt_loc.get('x', approx_world_x) - ego_loc['x']
                dy   = gt_loc.get('y', approx_world_y) - ego_loc['y']
                x_ego =  dx * math.cos(-rad) - dy * math.sin(-rad)
                y_ego =  dx * math.sin(-rad) + dy * math.cos(-rad)
                pos = {
                    'x':        round(x_ego, 3),
                    'y':        round(y_ego, 3),
                    'z':        round(gt_loc.get('z', 0.0), 3),
                    'yaw':      round(gt_match.get('yaw', 0.0), 3),
                    'speed_ms': round(gt_match.get('speed_ms', 0.0), 3),
                    'vx':       round(gt_match.get('velocity', {}).get('x', 0.0), 3),
                    'vy':       round(gt_match.get('velocity', {}).get('y', 0.0), 3),
                }
                gt_actor_type = gt_match.get('type', actor_type)
                actor_id_src  = gt_match.get('id', f'actor_{frame_id}_{i}')
            else:
                pos           = est_pos
                gt_actor_type = actor_type
                actor_id_src  = f'actor_{frame_id}_{i}'

            nodes_data.append({
                'actor_id':   f'{actor_id_src}',
                'actor_type': gt_actor_type,
                'yolo_class': class_name,
                'frame_id':   frame_id,
                'run_id':     run_id,
                'confidence': round(confidence, 4),
                'gt_fused':   gt_match is not None,
                **pos,
                'bbox_cx': round(bbox['x_center'], 4),
                'bbox_cy': round(bbox['y_center'], 4),
                'bbox_w':  round(bbox['width'],    4),
                'bbox_h':  round(bbox['height'],   4),
            })

        # Add GT actors missed by YOLO
        if gt_all_actors:
            detected_ids = {n['actor_id'] for n in nodes_data}
            rad = math.radians(ego_yaw)
            for gt_actor in gt_all_actors:
                aid = gt_actor.get('id', '')
                if aid in detected_ids:
                    continue
                gt_loc = gt_actor.get('loc', {})
                dx = gt_loc.get('x', 0) - ego_loc['x']
                dy = gt_loc.get('y', 0) - ego_loc['y']
                x_ego =  dx * math.cos(-rad) - dy * math.sin(-rad)
                y_ego =  dx * math.sin(-rad) + dy * math.cos(-rad)
                nodes_data.append({
                    'actor_id':   aid,
                    'actor_type': gt_actor.get('type', 'vehicle'),
                    'yolo_class': None,
                    'frame_id':   frame_id,
                    'run_id':     run_id,
                    'confidence': 0.0,
                    'gt_fused':   True,
                    'x':          round(x_ego, 3),
                    'y':          round(y_ego, 3),
                    'z':          round(gt_loc.get('z', 0.0), 3),
                    'yaw':        round(gt_actor.get('yaw', 0.0), 3),
                    'speed_ms':   round(gt_actor.get('speed_ms', 0.0), 3),
                    'vx':         round(gt_actor.get('velocity', {}).get('x', 0.0), 3),
                    'vy':         round(gt_actor.get('velocity', {}).get('y', 0.0), 3),
                    'bbox_cx': None, 'bbox_cy': None,
                    'bbox_w':  None, 'bbox_h':  None,
                })

        nodes_df = pd.DataFrame(nodes_data)
        edges_df = self._compute_edges(nodes_df)
        parsed_df = self._compute_frame_features(
            nodes_df     = nodes_df,
            edges_df     = edges_df,
            frame_id     = frame_id,
            run_id       = run_id,
            ego_speed_ms = ego_spd,
            target_label = target_label,
            gt_risk      = gt_risk,
            scenario     = scenario,
        )
        return nodes_df, edges_df, parsed_df

    def _compute_edges(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        if len(nodes_df) < 2:
            return pd.DataFrame(columns=['source_actor', 'target_actor', 'dist_m', 'ttc_s'])

        edges_data = []
        for i, ri in nodes_df.iterrows():
            for j, rj in nodes_df.iterrows():
                if i >= j:
                    continue
                dist_m = math.sqrt(
                    (ri['x'] - rj['x'])**2 +
                    (ri['y'] - rj['y'])**2 +
                    (ri['z'] - rj['z'])**2
                )
                rel_vx  = ri.get('vx', 0) - rj.get('vx', 0)
                rel_vy  = ri.get('vy', 0) - rj.get('vy', 0)
                dx, dy  = ri['x'] - rj['x'], ri['y'] - rj['y']
                closing = -(dx * rel_vx + dy * rel_vy) / max(dist_m, 0.1)
                ttc_s   = dist_m / closing if closing > 0.5 else 999.0

                edges_data.append({
                    'source_actor': ri['actor_id'],
                    'target_actor': rj['actor_id'],
                    'dist_m':       round(dist_m, 3),
                    'ttc_s':        round(ttc_s,  3),
                })
        return pd.DataFrame(edges_data)

    def _compute_frame_features(self,
                                nodes_df:     pd.DataFrame,
                                edges_df:     pd.DataFrame,
                                frame_id:     int,
                                run_id:       str,
                                ego_speed_ms: float,
                                target_label: int,
                                gt_risk:      Dict,
                                scenario:     str) -> pd.DataFrame:

        if len(nodes_df) > 0:
            distances          = np.sqrt(nodes_df['x']**2 +
                                         nodes_df['y']**2 +
                                         nodes_df['z']**2)
            min_dist_m         = float(distances.min())
            closest_actor_dist = min_dist_m
            num_threats        = int((distances < 20.0).sum())
        else:
            min_dist_m = closest_actor_dist = 999.0
            num_threats = 0

        # Prefer edge-based TTC when actors have real velocity data (GT-fused frames).
        # At live inference vx/vy are always 0, so edge closing speed is always <= 0
        # and every edge gets ttc=999. In that case fall back to ego-to-actor TTC:
        # closing_speed ≈ ego_speed_ms projected onto the forward (y) axis.
        if len(edges_df) > 0 and float(edges_df['ttc_s'].min()) < 998.0:
            ttc_min_s = float(edges_df['ttc_s'].min())
        elif len(nodes_df) > 0 and ego_speed_ms > 0.1:
            # y > 0 means the actor is ahead of ego in ego-frame coordinates
            ahead = nodes_df[nodes_df['y'] > 0.5].copy()
            if len(ahead) > 0:
                ttc_vals  = ahead['y'] / ego_speed_ms   # seconds until ego reaches actor
                ttc_min_s = float(ttc_vals.min())
            else:
                ttc_min_s = 999.0
        elif len(nodes_df) > 0:
            # ego is stopped but actors are present — use a 1 m/s reference speed
            # so TTC reflects proximity (e.g. 25 m away → ttc=25 s) rather than 999
            ahead = nodes_df[nodes_df['y'] > 0.5].copy()
            if len(ahead) > 0:
                ttc_vals  = ahead['y'] / 1.0
                ttc_min_s = float(ttc_vals.min())
            else:
                ttc_min_s = 999.0
        else:
            ttc_min_s = 999.0
        min_ttc = ttc_min_s

        gt_score    = gt_risk.get('score',      None)
        gt_ttc      = gt_risk.get('ttc_min_s',  None)
        gt_min_dist = gt_risk.get('min_dist_m', None)

        return pd.DataFrame([{
            'frame_id':           frame_id,
            'run_id':             run_id,
            'scenario':           scenario,
            'ego_speed_ms':       round(ego_speed_ms, 3),
            'min_dist_m':         min_dist_m,
            'ttc_min_s':          ttc_min_s,
            'num_threats':        num_threats,
            'min_ttc':            min_ttc,
            'closest_actor_dist': closest_actor_dist,
            'yolo_detections':    len(nodes_df[nodes_df['confidence'] > 0])
                                  if 'confidence' in nodes_df.columns else 0,
            'gt_fused_count':     int(nodes_df['gt_fused'].sum())
                                  if 'gt_fused' in nodes_df.columns else 0,
            'gt_risk_score':      gt_score,
            'gt_ttc_min_s':       gt_ttc,
            'gt_min_dist_m':      gt_min_dist,
            'target':             target_label,
        }])


# ──────────────────────────────────────────────────────────────────────────────
# LiveRiskPipeline  –  deployment wrapper for FastAPI
# ──────────────────────────────────────────────────────────────────────────────

# Feature order must exactly match what the XGBoost model was trained on.
XGB_FEATURES = [
    'ego_speed_ms',
    'min_dist_m',
    'num_threats',
    'closest_actor_dist',
    'yolo_detections',
    'gt_fused_count',
    'gt_min_dist_m',
    'distance_to_speed_ratio',
    'closest_actor_to_speed_ratio',
    'threat_density',
    'ego_kinetic_proxy',
    'space_compression_index',
    'log_space_compression',
    'distance_velocity',
    'speed_delta',
    'distance_rolling_var',
    'scenario_cut_in',
    'scenario_head_on_close',
    'scenario_normal',
    'scenario_ped_crossing',
    'scenario_ped_jaywalking',
    'scenario_sudden_brake',
]

RISK_LABELS = {0: 'safe', 1: 'medium', 2: 'high', 3: 'critical'}


class LiveRiskPipeline:
    """
    Deployment wrapper used by the FastAPI service.

    Responsibilities:
      1. Run PerceptionPipeline on each incoming frame to extract features.
      2. Fill in XGBoost feature columns, using YOLO-computed fallbacks when
         no ground-truth simulation metadata is available (i.e., live inference).
      3. Run the trained XGBoost classifier and return a risk prediction.
      4. Maintain a rolling speed history for the active WebSocket session
         (reset on each new connection via reset_history()).
    """

    def __init__(self, model_path: str = 'yolov8n.pt',
                 config_path: str = 'best_xgboost_model.json',
                 smoothing_window: int = 5,
                 hysteresis_frames: int = 3):
        """
        smoothing_window:   number of recent frames' probability vectors to
                             average together before picking a risk level.
        hysteresis_frames:  a NEW risk level must be the smoothed argmax for
                             this many consecutive frames before the publicly
                             reported risk_level actually switches. Prevents
                             single-frame blips from flipping the HUD label
                             even after averaging.
        """
        self.perception = PerceptionPipeline(model_path=model_path)

        # Load via the native Booster API.
        # XGBClassifier.load_model() does NOT restore sklearn-compat attributes
        # (n_classes_, classes_) so predict_proba() raises AttributeError.
        # Using xgb.Booster directly and calling predict() with output_margin=False
        # gives raw class probabilities without needing those attributes.
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(config_path)
        print(f"[LiveRiskPipeline] XGBoost model loaded from '{config_path}'")

        self._frame_counter = 0
        self._speed_history: List[float] = []

        # ── Temporal smoothing state ──────────────────────────────────────
        self.smoothing_window  = smoothing_window
        self.hysteresis_frames = hysteresis_frames
        self._proba_window: List[List[float]] = []   # recent per-frame probability vectors
        self._displayed_class:   Optional[int] = None  # what's currently shown to the user
        self._pending_class:     Optional[int] = None  # candidate class trying to take over
        self._pending_streak:    int = 0                # consecutive frames pending class has won

    # ---------------------------------------------------------------- public

    def predict_risk(self,
                     frame:        np.ndarray,
                     ego_speed_ms: float = 0.0,
                     scenario:     str   = "normal") -> Dict:
        """
        Run the full perception + classification pipeline on a single frame.

        Args:
            frame:        BGR numpy array (from cv2.imdecode / camera capture).
            ego_speed_ms: Current ego-vehicle speed in m/s.
            scenario:     Driving scenario tag (e.g. 'normal', 'highway').

        Returns:
            dict with keys: risk_level (str), risk_class (int),
                            probabilities (list[float]), num_detections (int),
                            closest_dist_m (float), ttc_s (float).
        """
        self._speed_history.append(ego_speed_ms)
        self._frame_counter += 1

        # No simulator metadata available at live inference time –
        # PerceptionPipeline gracefully falls back to YOLO-only heuristics.
        nodes_df, edges_df, parsed_df = self.perception.process_frame(
            frame        = frame,
            frame_id     = self._frame_counter,
            run_id       = "live",
            metadata     = None,     # no GT metadata during live inference
            target_label = None,
        )

        # ── Build the full 21-feature row expected by XGBoost ────────────────
        # When gt_min_dist_m is NaN (no simulator metadata), fall back to the
        # YOLO-computed closest_actor_dist so the column is never NaN at inference.
        row = parsed_df.iloc[0]

        min_dist_m         = float(row['min_dist_m'])
        closest_actor_dist = float(row['closest_actor_dist'])
        num_threats        = int(row['num_threats'])
        yolo_detections    = int(row.get('yolo_detections', 0))
        gt_fused_count     = int(row.get('gt_fused_count', 0))

        gt_min_dist = row.get('gt_min_dist_m')
        gt_min_dist_m = float(gt_min_dist) if pd.notna(gt_min_dist) \
                        else closest_actor_dist

        # ── Rolling speed history features ───────────────────────────────────
        spd_hist = self._speed_history          # already includes current frame
        # FIX: Maintain a parallel history list for distances if you want rolling distance variance
        if not hasattr(self, '_distance_history'):
            self._distance_history = []
        self._distance_history.append(min_dist_m)
        dist_hist = self._distance_history
        speed_delta          = (spd_hist[-1] - spd_hist[-2]) if len(spd_hist) >= 2 else 0.0
        distance_velocity = (dist_hist[-1] - dist_hist[-2]) if len(dist_hist) >= 2 else 0.0
        # FIX: Rolling variance over a 5-frame window on DISTANCE (matching training notebook)
        if len(dist_hist) >= 2:
            window = dist_hist[-5:]
            distance_rolling_var = float(np.var(window)) if len(window) > 1 else 0.0
        else:
            distance_rolling_var = 0.0
        # ── Derived ratio / physics features ─────────────────────────────────
        safe_speed = max(ego_speed_ms, 0.1)
        safe_dist  = max(min_dist_m,   0.1)

        distance_to_speed_ratio        = min_dist_m         / safe_speed
        closest_actor_to_speed_ratio   = closest_actor_dist / safe_speed
        threat_density                 = num_threats        / max(min_dist_m, 1.0)
        ego_kinetic_proxy              = 0.5 * ego_speed_ms ** 2
        space_compression_index        = ego_speed_ms       / safe_dist
        log_space_compression          = float(np.log1p(space_compression_index))
        # NOTE: distance_velocity is the rolling closing-rate computed above
        # (dist_hist[-1] - dist_hist[-2]) — do NOT overwrite it here.
        # distance_speed_product is a separate derived feature, kept distinct
        # so it doesn't clobber the closing-rate signal the model expects.
        distance_speed_product         = min_dist_m * ego_speed_ms

        # ── One-hot scenario encoding ─────────────────────────────────────────
        known_scenarios = [
            'scenario_cut_in',
            'scenario_head_on_close',
            'scenario_normal',
            'scenario_ped_crossing',
            'scenario_ped_jaywalking',
            'scenario_sudden_brake',
        ]
        # Normalise: lowercase + spaces→underscores so "ped crossing" == "ped_crossing"
        scenario_col = f'scenario_{scenario.lower().replace(" ", "_")}'
        scenario_ohe = {s: int(s == scenario_col) for s in known_scenarios}
 

        features = {
            'ego_speed_ms':                 ego_speed_ms,
            'min_dist_m':                   min_dist_m,
            'num_threats':                  num_threats,
            'closest_actor_dist':           closest_actor_dist,
            'yolo_detections':              yolo_detections,
            'gt_fused_count':               gt_fused_count,
            'gt_min_dist_m':                gt_min_dist_m,
            'distance_to_speed_ratio':      distance_to_speed_ratio,
            'closest_actor_to_speed_ratio': closest_actor_to_speed_ratio,
            'threat_density':               threat_density,
            'ego_kinetic_proxy':            ego_kinetic_proxy,
            'space_compression_index':      space_compression_index,
            'log_space_compression':        log_space_compression,
            'distance_velocity':            distance_velocity,
            'speed_delta':                  speed_delta,
            'distance_rolling_var':         distance_rolling_var,
            **scenario_ohe,
        }

        X = pd.DataFrame([features])[XGB_FEATURES]

        # xgb.Booster.predict() returns a (1, n_classes) probability matrix
        # for multi-class problems when the model was trained with softprob objective.
        dmatrix       = xgb.DMatrix(X)
        raw           = self.xgb_model.predict(dmatrix)          # shape: (1, 4)
        raw_proba     = raw[0].tolist() if raw.ndim == 2 else raw.tolist()
        raw_class     = int(np.argmax(raw_proba))

        # ── Temporal smoothing: average probabilities over a rolling window ──
        # so a single noisy frame can't flip the displayed risk level.
        self._proba_window.append(raw_proba)
        if len(self._proba_window) > self.smoothing_window:
            self._proba_window.pop(0)
        smoothed_proba = np.mean(self._proba_window, axis=0).tolist()
        smoothed_class = int(np.argmax(smoothed_proba))

        # ── Hysteresis: only switch the DISPLAYED level once the smoothed
        # class has been the leader for several consecutive frames. This
        # keeps the HUD label stable while still reacting within ~1s to a
        # genuine, sustained change (e.g. an actual emergency brake event).
        if self._displayed_class is None:
            self._displayed_class = smoothed_class
            self._pending_class, self._pending_streak = smoothed_class, 0
        elif smoothed_class == self._displayed_class:
            self._pending_class, self._pending_streak = self._displayed_class, 0
        else:
            if smoothed_class == self._pending_class:
                self._pending_streak += 1
            else:
                self._pending_class, self._pending_streak = smoothed_class, 1

            # Escalations to a MORE dangerous level are allowed through faster
            # (safety-critical — don't sit on a stale "safe" label), while
            # de-escalations require the full hysteresis streak to confirm
            # the danger has actually passed.
            required_streak = 1 if smoothed_class > self._displayed_class else self.hysteresis_frames
            if self._pending_streak >= required_streak:
                self._displayed_class = smoothed_class
                self._pending_class, self._pending_streak = self._displayed_class, 0

        # The publicly reported risk_level/risk_class MUST always match the
        # returned probabilities — otherwise the UI shows a label that
        # contradicts its own probability bars. Use the smoothed argmax
        # directly rather than the hysteresis-lagged displayed_class.
        risk_class = smoothed_class
        risk_level = RISK_LABELS[risk_class]
        proba      = smoothed_proba   # report the smoothed probabilities, not the raw single-frame ones

        # `stable_risk_level` is exposed separately for callers (e.g. a live
        # HUD) that want a flicker-resistant label. It intentionally can lag
        # behind `risk_level` for a few frames after a de-escalation — that's
        # the point of hysteresis. Never use it as the "headline" label.
        stable_risk_level = RISK_LABELS[self._displayed_class]

        # Build a lightweight detections list for the client to draw boxes
        detections_out = []
        if len(nodes_df) > 0:
            for _, nd in nodes_df.iterrows():
                if nd.get('bbox_cx') is None:
                    continue
                detections_out.append({
                    'actor_type': str(nd.get('actor_type', 'unknown')),
                    'confidence': round(float(nd.get('confidence', 0.0)), 3),
                    'bbox_cx':    round(float(nd['bbox_cx']), 4),
                    'bbox_cy':    round(float(nd['bbox_cy']), 4),
                    'bbox_w':     round(float(nd['bbox_w']),  4),
                    'bbox_h':     round(float(nd['bbox_h']),  4),
                })

        return {
            'risk_level':        risk_level,
            'risk_class':        risk_class,
            'stable_risk_level': stable_risk_level,  # hysteresis-smoothed, for HUD flicker suppression only
            'probabilities':  {RISK_LABELS[i]: round(p, 4)
                            for i, p in enumerate(proba)},
            'num_detections': int(row.get('yolo_detections', 0)),
            'closest_dist_m': round(float(row['closest_actor_dist']), 2),
            'ttc_s':          round(float(row['ttc_min_s']), 2),
            'scenario':       scenario,
            'frame_id':       self._frame_counter,
            'detections':     detections_out,   # ← NEW: bounding boxes for client HUD
        }

    def reset_history(self) -> None:
        """Flush rolling session state. Called on each new WebSocket connection."""
        self._frame_counter = 0
        self._speed_history = []
        self._distance_history = [] # Clear distance tracker
        self._proba_window = []
        self._displayed_class = None
        self._pending_class = None
        self._pending_streak = 0
        print("[LiveRiskPipeline] Session history reset.")