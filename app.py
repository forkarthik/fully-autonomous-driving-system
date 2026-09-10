import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Configure Path Resolution
ROOT_DIR = Path(__file__).resolve().parent
ADS_DIR = ROOT_DIR / "autonomous_driving_system"
for p in [str(ROOT_DIR), str(ADS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from autonomous_driving_system.perception.camera_geometry import CameraGeometry
from autonomous_driving_system.perception.collision_warning import CollisionWarning
from autonomous_driving_system.perception.lane_detector import LaneDetector
from autonomous_driving_system.perception.object_detector import ObjectDetector
from autonomous_driving_system.perception.pothole_detector import PotholeDetector
from autonomous_driving_system.planning.costmap_planner import CostmapPlanner
from autonomous_driving_system.control.stanley_controller import StanleyController
from autonomous_driving_system.utils.serial_controller import SerialController

# Import HUD rendering functions from main
from autonomous_driving_system.main import (
    draw_3d_trajectory_ribbon,
    draw_nextgen_detections,
    draw_topographical_potholes,
    draw_state_and_alerts,
    draw_cockpit_dashboard,
    draw_tactical_bev_radar,
    StateMachine
)

# Page Configuration
st.set_page_config(
    page_title="ADAS AI Copilot | Jetson Orin Nano",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Cyberpunk Styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stMetric {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 8px;
        padding: 10px 14px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }
    .stMetric label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #00f0ff !important;
        font-family: 'Courier New', monospace;
        font-weight: 700;
    }
    .hud-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        background: linear-gradient(90deg, #00f0ff, #7000ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .hud-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
        letter-spacing: 0.05em;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load perception models and planners once across app lifetime."""
    engine_candidates = [
        "yolo11n.engine",
        "/home/smvec-eee/Desktop/yolo11n.engine",
        str(ROOT_DIR / "yolo11n.engine"),
        "yolo11n.pt",
        "/home/smvec-eee/Desktop/yolo11n.pt"
    ]
    model_path = "yolo11n.pt"
    for cand in engine_candidates:
        if os.path.exists(cand):
            model_path = cand
            break

    obj_det = ObjectDetector(model_path=model_path, conf_thres=0.35)
    lane_det = LaneDetector()
    pothole_det = PotholeDetector()
    cam_geom = CameraGeometry(image_width=640, image_height=480, camera_height_m=1.2, pitch_deg=4.0)
    costmap_planner = CostmapPlanner(cam_geom)
    stanley_ctrl = StanleyController()

    return obj_det, lane_det, pothole_det, cam_geom, costmap_planner, stanley_ctrl


@st.cache_resource
def get_serial(port):
    if not port or port.strip().lower() in ("none", ""):
        return None
    try:
        sc = SerialController(port)
        sc.start()
        return sc
    except Exception as e:
        st.sidebar.warning(f"Serial not available on {port}: {e}")
        return None


def main():
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.markdown('<div class="hud-title">⚡ ADAS TACTICAL COPILOT</div>', unsafe_allow_html=True)
        st.caption("Jetson Orin Nano 8GB | JetPack 6 | 3D IPM + Follow-the-Gap + Stanley + Pothole Depth Estimation")
    with col_hdr2:
        st.markdown("""
        <div style="text-align: right; padding-top: 8px;">
            <span class="hud-badge" style="background: rgba(0, 240, 255, 0.15); color: #00f0ff; border: 1px solid #00f0ff;">ARM64 / MAXN</span>
            <span class="hud-badge" style="background: rgba(0, 255, 128, 0.15); color: #00ff80; border: 1px solid #00ff80;">TENSORRT READY</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Sidebar Configuration ──
    st.sidebar.title("⚙️ Mission Control")
    mode = st.sidebar.radio("Navigation Mode", ["Autonomous (F1TENTH / Stanley)", "Manual Override"])
    display_mode = st.sidebar.selectbox("HUD Render Pipeline", ["Cyberpunk Swaayatt HUD", "Diagnostics & Masks", "BEV Radar Focus"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Hardware Peripherals")
    source_choice = st.sidebar.selectbox("Video Source", ["Test Static Image (test.jpg)", "Camera Device (/dev/video0 or 0)", "Custom Path"])
    
    if source_choice == "Custom Path":
        camera_source = st.sidebar.text_input("Custom Video / Device Path", "0")
    elif source_choice == "Camera Device (/dev/video0 or 0)":
        camera_source = st.sidebar.text_input("Camera Index / Path", "0")
    else:
        camera_source = "test.jpg"

    serial_port = st.sidebar.text_input("ESP32 Serial Port", "/dev/ttyUSB0")
    enable_serial = st.sidebar.checkbox("Enable Serial Commands", value=False)
    
    serial = None
    if enable_serial:
        serial = get_serial(serial_port)

    # Manual Control Sliders
    if mode == "Manual Override":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🕹️ Manual Actuator")
        man_angle = st.sidebar.slider("Steering Angle (deg)", 0, 180, 90)
        man_speed = st.sidebar.slider("Drive Throttle (PWM)", -255, 255, 0)
        if serial and st.sidebar.button("Send Manual Command"):
            serial.send_command(man_angle, man_speed, alarm=0)

    # Load System Engine
    try:
        obj_det, lane_det, pothole_det, cam_geom, costmap_planner, stanley_ctrl = load_models()
    except Exception as e:
        st.error(f"Error initializing perception pipeline: {e}")
        return

    cw = CollisionWarning((480, 640), camera_geom=cam_geom)
    sm = StateMachine()

    # Layout Columns
    feed_col, telem_col = st.columns([2.2, 1])

    with telem_col:
        st.subheader("📊 Live Telemetry")
        m_status = st.metric("State Machine", "INIT", "Standing By")
        c1, c2 = st.columns(2)
        with c1:
            m_speed = st.metric("Ground Speed", "0.0 KM/H")
            m_steer = st.metric("Steering Law", "90.0°")
        with c2:
            m_fps = st.metric("Throughput", "0.0 FPS")
            m_potholes = st.metric("Potholes", "0")

        st.markdown("##### Threat Alert Stack")
        alert_box = st.empty()

        st.markdown("##### Tactical Radar")
        radar_window = st.empty()

    with feed_col:
        run_btn = st.checkbox("Engage AI Pipeline", value=True)
        frame_window = st.image([])

    if not run_btn:
        st.info("Pipeline paused. Check 'Engage AI Pipeline' to run.")
        return

    # Video Loop / Image Processor
    is_static = (source_choice == "Test Static Image (test.jpg)")
    if is_static:
        test_img_path = str(ROOT_DIR / "test.jpg")
        if not os.path.exists(test_img_path):
            # Create synthetic test frame
            sample_frame = np.full((480, 640, 3), 40, dtype=np.uint8)
            cv2.putText(sample_frame, "No test.jpg found", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 240, 255), 2)
        else:
            sample_frame = cv2.imread(test_img_path)
            sample_frame = cv2.resize(sample_frame, (640, 480))
        frames_to_process = [sample_frame]
    else:
        src = int(camera_source) if camera_source.isdigit() else camera_source
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frames_to_process = None

    throttle = 0.0
    steering_angle = 90.0
    frame_idx = 0

    while run_btn:
        t_start = time.perf_counter()

        if is_static:
            frame = sample_frame.copy()
            time.sleep(0.08) # simulate frame rate
        else:
            ret, frame = cap.read()
            if not ret:
                st.warning("Video stream ended or camera disconnected.")
                break
            frame = cv2.resize(frame, (640, 480))

        frame_idx += 1

        # ── Pipeline Processing ──
        road_result = lane_det.detect(frame)
        potholes = pothole_det.detect(frame)
        detections = obj_det.detect(frame)
        warnings, roi_color = cw.check_collision(detections, frame)

        active_tracks = cw.tracker.tracks
        plan_result = costmap_planner.plan(
            road_result,
            active_tracks,
            potholes,
            current_speed=max(0.5, abs(throttle) * 0.035)
        )

        state = sm.update(warnings, plan_result, potholes)

        if mode.startswith("Autonomous"):
            target_lat = plan_result.get('target_lateral_offset_m', 0.0)
            target_heading = plan_result.get('target_angle_deg', 0.0)
            if state in ("RED LIGHT", "STOP SIGN", "EMERGENCY AEB", "BRAKING"):
                target_lat, target_heading = 0.0, 0.0

            steering_angle = stanley_ctrl.compute_steering(
                target_lat,
                target_heading,
                current_speed_ms=max(0.6, abs(throttle) * 0.035)
            )
            throttle = stanley_ctrl.compute_throttle(state, steering_angle)

            if serial and enable_serial:
                alarm_level = 2 if "AEB" in state else (1 if ("CROSSING" in state or "WARN" in state) else 0)
                serial.send_command(steering_angle, throttle, alarm=alarm_level)

        dt = time.perf_counter() - t_start
        fps = 1.0 / max(dt, 0.001)

        # ── Visual Rendering ──
        vis = frame.copy()
        if display_mode == "Cyberpunk Swaayatt HUD":
            draw_3d_trajectory_ribbon(vis, road_result, plan_result, cam_geom, steering_angle)
            draw_nextgen_detections(vis, detections, active_tracks, cam_geom, warnings)
            draw_topographical_potholes(vis, potholes)
            draw_state_and_alerts(vis, state, warnings)
            num_det = len(detections.boxes) if detections and detections.boxes is not None else 0
            draw_cockpit_dashboard(vis, fps, num_det, len(potholes), state, steering_angle, throttle)
            draw_tactical_bev_radar(vis, plan_result)
        elif display_mode == "Diagnostics & Masks":
            road_mask = road_result.get('road_mask')
            if road_mask is not None and np.any(road_mask > 0):
                overlay = vis.copy()
                overlay[road_mask > 0] = [0, 220, 0]
                cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
            draw_nextgen_detections(vis, detections, active_tracks, cam_geom, warnings)
            draw_topographical_potholes(vis, potholes)
            draw_tactical_bev_radar(vis, plan_result)
        else: # BEV Radar Focus
            radar_img = plan_result.get('costmap_bev')
            if radar_img is not None:
                vis = cv2.resize(radar_img, (640, 480), interpolation=cv2.INTER_NEAREST)

        # Display Live RGB Feed in Streamlit
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        frame_window.image(vis_rgb, use_container_width=True)

        # Update Telemetry Metrics
        speed_kmh = abs(throttle) * 0.035 * 3.6
        m_status.metric("State Machine", state, f"Action: {'AEB BRAKE' if 'AEB' in state else 'CORRIDOR FOLLOW'}")
        m_speed.metric("Ground Speed", f"{speed_kmh:.1f} KM/H", f"PWM: {int(throttle)}")
        m_steer.metric("Steering Law", f"{steering_angle:.1f}°", f"Err: {plan_result.get('target_angle_deg', 0.0):+.1f}°")
        m_fps.metric("Throughput", f"{fps:.1f} FPS", f"{dt*1000.0:.1f} ms")
        m_potholes.metric("Potholes", f"{len(potholes)} Detected")

        if warnings:
            alert_box.error("\n".join(warnings))
        else:
            alert_box.success("✅ Path Nominal - No Imminent Trajectory Hazards")

        # Show standalone BEV radar in telemetry column
        bev_img = plan_result.get('costmap_bev')
        if bev_img is not None:
            radar_rgb = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
            radar_window.image(radar_rgb, caption="Bird's-Eye View (10m x 4m Local Grid)", use_container_width=True)

        if is_static:
            break

    if not is_static and 'cap' in locals() and cap.isOpened():
        cap.release()


if __name__ == "__main__":
    main()
