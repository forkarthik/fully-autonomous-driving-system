import streamlit as st
import cv2
import numpy as np
import time
import queue
import threading
from perception.object_detector import ObjectDetector
from perception.lane_detector import LaneDetector
from perception.collision_warning import CollisionWarning
from utils.serial_controller import SerialController

# Configure page
st.set_page_config(page_title="ADAS Control Panel", layout="wide")

# --- CACHED RESOURCES ---
@st.cache_resource
def get_detectors():
    print("Loading Models...")
    # Try TensorRT first
    try:
        obj = ObjectDetector(model_path='/home/smvec-eee/Desktop/yolo11n.engine')
        print("Loaded TensorRT Engine")
    except:
        obj = ObjectDetector(model_path='/home/smvec-eee/Desktop/yolo11n.pt')
        print("Loaded PyTorch Model")
        
    lane = LaneDetector()
    return obj, lane

@st.cache_resource
def get_serial_controller(port='/dev/ttyUSB0'):
    sc = SerialController(port)
    sc.start()
    return sc

# --- APP LOGIC ---
def main():
    st.title("🚗 ADAS Control Panel")
    
    # Sidebar Controls
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Mode", ["Manual", "Autonomous"])
    
    # Connection Strings
    serial_port = st.sidebar.text_input("Serial Port", "/dev/ttyUSB0")
    camera_source = st.sidebar.text_input("Camera Source", "0")
    
    # Initialize Resources
    try:
        obj_det, lane_det = get_detectors()
        serial = get_serial_controller(serial_port)
        st.sidebar.success(f"Connected to {serial_port}")
    except Exception as e:
        st.sidebar.error(f"Initialization Failed: {e}")
        return

    # Manual Control Widgets
    st.sidebar.markdown("---")
    st.sidebar.subheader("Manual Control")
    # Using session state to persist slider values
    if 'speed' not in st.session_state: st.session_state.speed = 0
    if 'angle' not in st.session_state: st.session_state.angle = 90
    
    # Callback to send Manual Commands instantly
    def send_manual_cmd():
        if mode == "Manual":
            serial.send_command(st.session_state.angle, st.session_state.speed)

    speed = st.sidebar.slider("Speed", -255, 255, key="speed", on_change=send_manual_cmd)
    angle = st.sidebar.slider("Steering Angle", 0, 180, key="angle", on_change=send_manual_cmd)
    
    # --- MAIN VIEW ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Live Feed")
        run_btn = st.checkbox("Start Camera", value=True)
        frame_window = st.image([])
    
    with col2:
        st.header("Telemetry")
        fps_text = st.empty()
        status_text = st.empty()
        warning_box = st.empty()

    # --- LOOP ---
    if run_btn:
        cap = cv2.VideoCapture(int(camera_source) if camera_source.isdigit() else camera_source)
        
        # Collision Warning Instance (Per run)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cw = CollisionWarning((height, width))
        
        while run_btn:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture frame")
                break
                
            start_time = time.time()
            
            # --- PROCESSING ---
            # 1. Perception
            detections = obj_det.detect(frame)
            lane_img, lanes_info = lane_det.detect(frame)
            
            # 2. Collision Check
            warnings, roi_color = cw.check_collision(detections, frame)
            
            # 3. Autonomous Logic
            if mode == "Autonomous":
                # Steering
                steering_angle = 90
                target_x = width // 2
                if lanes_info.get('left_x') and lanes_info.get('right_x'):
                    target_x = (lanes_info['left_x'] + lanes_info['right_x']) // 2
                elif lanes_info.get('left_x'): target_x = lanes_info['left_x'] + 300
                elif lanes_info.get('right_x'): target_x = lanes_info['right_x'] - 300
                    
                error = target_x - (width // 2)
                steering_angle = max(0, min(180, 90 + int(error * 0.15)))
                
                # Throttle
                throttle = 60
                stop = False
                for w in warnings:
                    if "CRITICAL" in w or "RED LIGHT" in w:
                        stop = True
                if stop: throttle = 0
                
                # Send
                serial.send_command(steering_angle, throttle)
                status_text.metric("Status", "AUTO", f"Steer: {steering_angle} | Speed: {throttle}")
            else:
                status_text.metric("Status", "MANUAL", f"Steer: {st.session_state.angle} | Speed: {st.session_state.speed}")

            # --- VISUALIZATION ---
            # Draw Lanes
            if lane_img is not None:
                # Convert 3-channel lane image to 2D mask
                mask = np.any(lane_img > 0, axis=-1)
                frame[mask] = [0, 255, 0] # Simple Green overlay
                
            # Draw Objects
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                label = f"{detections.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw ROI
            if roi_color == (0, 0, 255): # Critical
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)

            # Convert to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)
            
            # FPS
            fps = 1.0 / (time.time() - start_time)
            fps_text.metric("FPS", f"{fps:.1f}")
            
            # Warnings
            if warnings:
                warning_box.error(f"WARNINGS: {warnings}")
            else:
                warning_box.success("Path Clear")

        cap.release()

if __name__ == "__main__":
    main()
