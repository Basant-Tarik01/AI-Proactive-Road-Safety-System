import os
import sys
import tempfile

# ── Path resolution (works locally and in Docker) ─────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(ROOT_DIR, "road_safety_pipline")
APP_DIR      = os.path.join(PIPELINE_DIR, "app")
if not os.path.isdir(APP_DIR):
    APP_DIR = os.path.join(ROOT_DIR, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
MODEL_DIR = PIPELINE_DIR if os.path.isdir(PIPELINE_DIR) else ROOT_DIR

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from model import LiveRiskPipeline, RISK_LABELS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AV Risk Perception",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Risk colour map ───────────────────────────────────────────────────────────
RISK_COLORS = {
    "safe":     ("#38ef7d", "rgba(56,239,125,0.08)"),
    "medium":   ("#f6d365", "rgba(246,211,101,0.08)"),
    "high":     ("#fa8231", "rgba(250,130,49,0.08)"),
    "critical": ("#ff4757", "rgba(255,71,87,0.08)"),
}
PROB_COLORS = {
    "safe": "#38ef7d", "medium": "#f6d365",
    "high": "#fa8231", "critical": "#ff4757",
}
SCENARIO_LABELS = {
    "normal":         "🟢 Normal",
    "cut_in":         "⚡ Cut-In",
    "head_on_close":  "💥 Head-On Close",
    "ped_crossing":   "🚶 Ped Crossing",
    "ped_jaywalking": "🏃 Ped Jaywalking",
    "sudden_brake":   "🛑 Sudden Brake",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

#MainMenu, footer, header { visibility: hidden; }

* { font-family: 'Inter', sans-serif !important; }

/* ── background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0d1f3c 0%, #060b14 50%, #080e1c 100%);
    color: #e2e8f0;
}

/* ── left control column styling ── */
[data-testid="stSidebar"] { display: none; }

/* ── metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0f1e35 0%, #0a1525 100%);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(56,189,248,0.28);
    box-shadow: 0 4px 32px rgba(56,189,248,0.08);
}
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(56,189,248,0.08) 0%, rgba(99,102,241,0.08) 100%);
    color: #7dd3fc;
    border: 1px solid rgba(56,189,248,0.22);
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.84rem;
    letter-spacing: 0.04em;
    padding: 10px 20px;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1);
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(56,189,248,0.16) 0%, rgba(99,102,241,0.16) 100%);
    border-color: rgba(56,189,248,0.5);
    box-shadow: 0 0 20px rgba(56,189,248,0.15), 0 4px 12px rgba(0,0,0,0.2);
    transform: translateY(-2px);
    color: #bae6fd;
}

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(15,30,53,0.6) 0%, rgba(10,21,37,0.8) 100%);
    border: 2px dashed rgba(56,189,248,0.18);
    border-radius: 16px;
    padding: 12px;
    transition: border-color 0.25s, box-shadow 0.25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(56,189,248,0.4);
    box-shadow: 0 0 24px rgba(56,189,248,0.06);
}
/* Fix doubled upload button text */
[data-testid="stFileUploader"] button span:last-child { display: none; }
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, rgba(56,189,248,0.1), rgba(129,140,248,0.1)) !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 8px !important;
    color: #7dd3fc !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, rgba(56,189,248,0.18), rgba(129,140,248,0.18)) !important;
    border-color: rgba(56,189,248,0.5) !important;
}

/* ── dividers ── */
hr { border: none; border-top: 1px solid rgba(56,189,248,0.07); margin: 16px 0; }

/* ── risk badge ── */
.risk-badge {
    display: block;
    text-align: center;
    padding: 16px 40px;
    border-radius: 100px;
    font-size: 1.6rem;
    font-weight: 900;
    letter-spacing: 4px;
    margin: 4px auto 24px;
    width: fit-content;
    animation: glow-pulse 2.4s ease-in-out infinite;
    position: relative;
}
.risk-badge::before {
    content: '';
    position: absolute;
    inset: -2px;
    border-radius: 100px;
    opacity: 0.3;
    filter: blur(8px);
    background: inherit;
    z-index: -1;
}
@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 12px currentColor, 0 0 32px rgba(0,0,0,0.5); transform: scale(1); }
    50%       { box-shadow: 0 0 28px currentColor, 0 0 48px rgba(0,0,0,0.3); transform: scale(1.03); }
}

/* ── probability bars ── */
.prob-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }
.prob-label { font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.10em; font-weight:600; }
.prob-pct   { font-size:0.75rem; font-weight:700; }
.bar-track  {
    background: rgba(255,255,255,0.04);
    border-radius: 100px;
    height: 8px;
    margin-bottom: 14px;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
}
.bar-fill { height:100%; border-radius:100px; box-shadow: 0 0 8px currentColor; }

/* ── detection table ── */
.det-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.det-table th {
    color: #334155;
    text-transform: uppercase;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    font-weight: 700;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(56,189,248,0.08);
}
.det-table td { padding: 9px 12px; border-bottom: 1px solid rgba(255,255,255,0.03); color: #cbd5e1; }
.det-table tr:hover td {
    background: rgba(56,189,248,0.04);
    color: #e2e8f0;
}

/* ── section titles ── */
.sec {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #38bdf8;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-weight: 700;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(56,189,248,0.10);
}

/* ── ctrl panel card (left column) ── */
.ctrl-card {
    background: linear-gradient(160deg, #0f1e35 0%, #0a1525 100%);
    border: 1px solid rgba(56,189,248,0.10);
    border-radius: 18px;
    padding: 20px 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 12px;
}
.ctrl-logo {
    text-align: center;
    padding: 8px 0 20px;
    border-bottom: 1px solid rgba(56,189,248,0.08);
    margin-bottom: 20px;
}
.ctrl-logo .emoji { font-size: 2.6rem; display: block; margin-bottom: 8px; }
.ctrl-logo .name  {
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.ctrl-logo .sub   { font-size: 0.6rem; color: #1e3a5f; letter-spacing: 0.08em; margin-top: 3px; }

/* ── segmented control ── */
[data-testid="stSegmentedControl"] {
    background: rgba(10,18,32,0.95) !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    border-radius: 100px !important;
    padding: 3px !important;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.3) !important;
}
[data-testid="stSegmentedControl"] button {
    border-radius: 100px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #334155 !important;
    border: none !important;
    padding: 8px 22px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: linear-gradient(135deg, #0f2d4a, #1a1f3e) !important;
    color: #7dd3fc !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    box-shadow: 0 2px 12px rgba(56,189,248,0.15), inset 0 1px 0 rgba(255,255,255,0.06) !important;
}

/* ── video progress ── */
.video-progress {
    background: rgba(255,255,255,0.04);
    border-radius: 100px;
    height: 5px;
    margin: 10px 0 18px;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.4);
}
.video-progress-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    box-shadow: 0 0 10px rgba(56,189,248,0.5);
    transition: width 0.25s ease;
}

/* ── slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    border: 2px solid rgba(255,255,255,0.2) !important;
    box-shadow: 0 0 10px rgba(56,189,248,0.4) !important;
}

/* ── selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: linear-gradient(135deg, #0f1e35, #0a1525) !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    border-radius: 10px !important;
}

/* ── frame counter badge ── */
.frame-badge {
    text-align: center;
    margin-top: 16px;
    padding: 10px;
    background: rgba(56,189,248,0.04);
    border: 1px solid rgba(56,189,248,0.08);
    border-radius: 12px;
}
.frame-badge .label { font-size: 0.62rem; color: #334155; text-transform: uppercase; letter-spacing: 0.12em; }
.frame-badge .value { font-size: 1.6rem; font-weight: 800; background: linear-gradient(90deg,#38bdf8,#818cf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
</style>
""", unsafe_allow_html=True)

# ── Load pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return LiveRiskPipeline(
        model_path=os.path.join(MODEL_DIR, "yolov8n.pt"),
        config_path=os.path.join(MODEL_DIR, "best_xgboost_model.json"),
    )

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"❌ Failed to initialise models: {e}")
    st.stop()

# Session state
if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "image"

# ── Top header (full width) ───────────────────────────────────────────────────
st.markdown("""
<div style='padding:24px 0 8px; display:flex; align-items:center; gap:16px;'>
    <div>
        <div style='
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            line-height: 1.1;
        '>Live Autonomous Risk Perception</div>
        <div style='color:#334155; font-size:0.8rem; margin-top:5px; letter-spacing:0.04em;'>
             &nbsp;YOLOv8 Detection &nbsp;·&nbsp;  &nbsp;XGBoost Classification &nbsp;·&nbsp;  &nbsp;Real-time
        </div>
    </div>
</div>
<div style='height:1px; background:linear-gradient(90deg, rgba(56,189,248,0.3), rgba(129,140,248,0.3), transparent); margin:12px 0 20px;'></div>
""", unsafe_allow_html=True)

# ── 3-column layout: [controls | content | risk panel] ───────────────────────
col_ctrl, col_main, col_risk = st.columns([1.2, 2.8, 1.8], gap="medium")

# ════════════════════════════════════
# LEFT COLUMN — Controls
# ════════════════════════════════════
with col_ctrl:
    st.markdown("""
    <div class="ctrl-card">
        <div class="ctrl-logo">
            <span class="emoji">🚗</span>
            <div class="name">AV RISK ENGINE</div>
            <div class="sub">Road Safety Perception System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec">⚙️ &nbsp;Simulation</div>', unsafe_allow_html=True)
    ego_speed_kmh = st.slider("Speed (km/h)", 0.0, 120.0, 40.0, 1.0)
    ego_speed_ms  = ego_speed_kmh / 3.6
    scenario = st.selectbox(
        "Scenario",
        options=list(SCENARIO_LABELS.keys()),
        format_func=lambda x: SCENARIO_LABELS[x],
    )

    st.markdown("---")
    st.markdown('<div class="sec">🎬 &nbsp;Video</div>', unsafe_allow_html=True)
    frame_skip = st.slider("Every N frames", 1, 10, 3,
                           help="Higher = faster, fewer frames analysed")
    max_frames = st.slider("Max frames", 10, 300, 60)

    st.markdown("---")

    if st.button("🔄  Reset History", use_container_width=True):
        pipeline.reset_history()
        st.session_state.reset_count += 1
        st.toast("✅  Pipeline history cleared!")

    st.markdown(f"""
    <div class="frame-badge">
        <div class="label">Frames Processed</div>
        <div class="value">{pipeline._frame_counter}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Mode switcher — segmented pill control ────────────────────────────────────
with col_main:
    selected_tab = st.segmented_control(
        "Mode",
        options=["📷  Image", "🎬  Video"],
        default="📷  Image",
        label_visibility="collapsed",
        key="tab_ctrl",
    )
    st.session_state.active_tab = "image" if selected_tab and "Image" in selected_tab else "video"
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: render one frame's results
# ══════════════════════════════════════════════════════════════════════════════
def annotate_frame(frame, nodes_df):
    """Draw bounding boxes on a copy of frame. Returns BGR annotated frame."""
    out = frame.copy()
    h, w, _ = out.shape
    if not nodes_df.empty:
        for _, row in nodes_df.iterrows():
            if pd.notna(row.get("bbox_cx")) and pd.notna(row.get("bbox_cy")):
                cx = row["bbox_cx"] * w;  cy = row["bbox_cy"] * h
                bw = row["bbox_w"]  * w;  bh = row["bbox_h"]  * h
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
                cv2.rectangle(out, (x1, y1), (x2, y2), (99, 179, 237), 2)
                label = f"{row.get('yolo_class','?')} {row.get('confidence',0):.2f}"
                cv2.putText(out, label, (x1, max(y1 - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (99, 179, 237), 1)
    return out


def render_risk_panel(prediction, nodes_df, scenario, ego_speed_kmh):
    """Render the right-side risk assessment panel."""
    risk_level = prediction["risk_level"].lower()
    color, bg  = RISK_COLORS.get(risk_level, ("#e2e8f0", "rgba(255,255,255,0.06)"))
    icons      = {"safe": "✅", "medium": "⚡", "high": "🔥", "critical": "🚨"}
    icon       = icons.get(risk_level, "❓")

    st.markdown('<div class="sec">🛡️ &nbsp;Risk Assessment</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="risk-badge" style="
        color:{color};
        background: linear-gradient(135deg, {bg}, rgba(0,0,0,0.3));
        border: 2px solid {color};
    ">{icon}&nbsp; {risk_level.upper()}</div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    m1.metric("🎯 Risk Class",    f"Class {prediction['risk_class']}")
    m2.metric("👁️ Detections",   f"{prediction['num_detections']} obj")
    m3, m4 = st.columns(2)
    m3.metric("📏 Closest",       f"{prediction['closest_dist_m']} m")
    m4.metric("⏱️ TTC",           f"{prediction['ttc_s']} s")

    st.markdown("---")
    st.markdown('<div class="sec">📊 &nbsp;Probabilities</div>', unsafe_allow_html=True)

    for r_name, r_prob in prediction["probabilities"].items():
        pct    = r_prob * 100
        pcolor = PROB_COLORS.get(r_name, "#38bdf8")
        st.markdown(f"""
        <div class="prob-row">
            <span class="prob-label">{r_name}</span>
            <span class="prob-pct" style="color:{pcolor};">{pct:.1f}%</span>
        </div>
        <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%; background:linear-gradient(90deg,{pcolor},{pcolor}88);"></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style='display:flex; flex-wrap:wrap; gap:6px; justify-content:center;'>
        <span style='background:rgba(56,189,248,0.07); border:1px solid rgba(56,189,248,0.18);
                     color:#38bdf8; font-weight:600; padding:4px 12px; border-radius:100px; font-size:0.72rem;'>
            {SCENARIO_LABELS.get(scenario, scenario)}
        </span>
        <span style='background:rgba(129,140,248,0.07); border:1px solid rgba(129,140,248,0.18);
                     color:#a5b4fc; font-weight:600; padding:4px 12px; border-radius:100px; font-size:0.72rem;'>
            🚀 {ego_speed_kmh:.0f} km/h
        </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "image":
    with col_main:
        uploaded_img = st.file_uploader(
            "Frame image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"img_uploader_{st.session_state.reset_count}",
        )

        if uploaded_img is None:
            st.markdown("""
            <div style='text-align:center; padding:48px 0; color:#2d3748;'>
                <div style='font-size:3rem; margin-bottom:12px;'>📷</div>
                <div style='font-size:0.95rem; color:#4a5568;'>Upload a road scene image</div>
                <div style='font-size:0.75rem; margin-top:6px;'>JPG · JPEG · PNG · max 200 MB</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                st.error("❌ Could not decode image.")
            else:
                with st.spinner("Running perception pipeline…"):
                    # A single uploaded image is an independent sample, not a
                    # continuation of whatever was analysed before it (another
                    # image, or a video run). Reset the rolling smoothing/
                    # hysteresis state first so leftover history can't leak in
                    # and make the label disagree with this image's own
                    # probabilities.
                    pipeline.reset_history()
                    prediction     = pipeline.predict_risk(frame, ego_speed_ms, scenario)
                    frame_id       = pipeline._frame_counter
                    nodes_df, _, _ = pipeline.perception.process_frame(
                        frame, frame_id=frame_id, run_id="dashboard_img"
                    )

                annotated   = annotate_frame(frame, nodes_df)
                rgb_preview = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                st.markdown('<div class="sec">📹 Perception Output</div>', unsafe_allow_html=True)
                st.image(rgb_preview, use_container_width=True)

                if not nodes_df.empty and "confidence" in nodes_df.columns:
                    visible = nodes_df[nodes_df["confidence"] > 0][
                        ["yolo_class", "actor_type", "confidence"]
                    ]
                    if not visible.empty:
                        st.markdown("---")
                        st.markdown('<div class="sec"> Detected Objects</div>', unsafe_allow_html=True)
                        rows_html = "".join(
                            f"<tr><td>{r['yolo_class']}</td>"
                            f"<td style='color:#718096;'>{r['actor_type']}</td>"
                            f"<td style='color:#68d391;font-weight:600;'>{r['confidence']:.0%}</td></tr>"
                            for _, r in visible.iterrows()
                        )
                        st.markdown(f"""
                        <table class="det-table">
                          <thead><tr><th>Class</th><th>Type</th><th>Conf</th></tr></thead>
                          <tbody>{rows_html}</tbody>
                        </table>""", unsafe_allow_html=True)

                with col_risk:
                    render_risk_panel(prediction, nodes_df, scenario, ego_speed_kmh)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "video":
    with col_main:
        uploaded_vid = st.file_uploader(
            "Video file",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
            key=f"vid_uploader_{st.session_state.reset_count}",
        )

        if uploaded_vid is None:
            st.markdown("""
            <div style='text-align:center; padding:48px 0; color:#2d3748;'>
                <div style='font-size:3rem; margin-bottom:12px;'>🎬</div>
                <div style='font-size:0.95rem; color:#4a5568;'>Upload a road scene video</div>
                <div style='font-size:0.75rem; margin-top:6px;'>MP4 · AVI · MOV · MKV · max 200 MB</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            suffix = os.path.splitext(uploaded_vid.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = tmp.name

            cap          = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = cap.get(cv2.CAP_PROP_FPS) or 25
            cap.release()

            st.markdown(f"""
            <div style='color:#718096; font-size:0.8rem; margin-bottom:12px;'>
                📽️ &nbsp;<span style='color:#90cdf4;'>{uploaded_vid.name}</span>
                &nbsp;·&nbsp; {total_frames} frames &nbsp;·&nbsp; {fps:.0f} fps
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶️  Run Analysis", key="run_video", use_container_width=True):
                pipeline.reset_history()
                cap = cv2.VideoCapture(tmp_path)

                st.markdown('<div class="sec">📹 Live Perception Feed</div>', unsafe_allow_html=True)
                frame_display = st.empty()
                progress_bar  = st.empty()
                frame_info    = st.empty()

                panel_placeholder = col_risk.empty()

                risk_history   = []
                frames_done    = 0
                frames_to_proc = min(max_frames, total_frames)
                frame_idx      = 0

                while True:
                    ret, frame = cap.read()
                    if not ret or frames_done >= frames_to_proc:
                        break

                    frame_idx += 1
                    if frame_idx % frame_skip != 0:
                        continue

                    prediction     = pipeline.predict_risk(frame, ego_speed_ms, scenario)
                    fid            = pipeline._frame_counter
                    nodes_df, _, _ = pipeline.perception.process_frame(
                        frame, frame_id=fid, run_id="dashboard_vid"
                    )

                    annotated  = annotate_frame(frame, nodes_df)
                    risk_level = prediction["risk_level"].lower()
                    color, _   = RISK_COLORS.get(risk_level, ("#e2e8f0", ""))

                    cv2.putText(annotated,
                                f"RISK: {risk_level.upper()}",
                                (12, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0,2,4))[::-1],
                                2)

                    frame_display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                        use_container_width=True)

                    pct = (frames_done + 1) / frames_to_proc * 100
                    progress_bar.markdown(f"""
                    <div class="video-progress">
                        <div class="video-progress-fill" style="width:{pct:.1f}%;"></div>
                    </div>""", unsafe_allow_html=True)
                    frame_info.markdown(
                        f"<div style='color:#4a5568;font-size:0.75rem;'>Frame {frames_done+1} / {frames_to_proc}</div>",
                        unsafe_allow_html=True)

                    with panel_placeholder.container():
                        render_risk_panel(prediction, nodes_df, scenario, ego_speed_kmh)

                    risk_history.append({
                        "frame": frames_done + 1,
                        "risk_class": prediction["risk_class"],
                        "risk_level": risk_level,
                    })
                    frames_done += 1

                cap.release()
                os.unlink(tmp_path)

                # ── Summary ───────────────────────────────────────────────────
                st.markdown("---")
                st.markdown('<div class="sec">📈 Session Summary</div>', unsafe_allow_html=True)

                if risk_history:
                    df_hist = pd.DataFrame(risk_history)
                    counts  = df_hist["risk_level"].value_counts()
                    s1, s2, s3, s4 = st.columns(4)
                    for col_s, level, icon in [
                        (s1, "safe",     "✅"),
                        (s2, "medium",   "⚠️"),
                        (s3, "high",     "🔶"),
                        (s4, "critical", "🚨"),
                    ]:
                        col_s.metric(f"{icon} {level.capitalize()}", f"{counts.get(level, 0)} frames")

                    st.markdown("**Risk class over time:**")
                    st.line_chart(df_hist.set_index("frame")["risk_class"])