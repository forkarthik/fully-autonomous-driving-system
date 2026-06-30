import cv2
import argparse
import numpy as np
from perception.object_detector import ObjectDetector
from perception.lane_detector import LaneDetector
from perception.collision_warning import CollisionWarning
from perception.pothole_detector import PotholeDetector
from utils.serial_controller import SerialController

import threading
import queue
import time
import os


# ═══════════════════════════════════════════════════════════════════════════
#  Worker Threads (unchanged — these are clean)
# ═══════════════════════════════════════════════════════════════════════════

class DetectionThread(threading.Thread):
    def __init__(self, detector, collision_warning, imgsz, device):
        super().__init__(daemon=True)
        self.detector = detector
        self.collision_warning = collision_warning
        self.imgsz = imgsz
        self.device = device
        self.running = True
        self.input_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.latest_result = None

    def detect_async(self, frame):
        if self.input_queue.full():
            try: self.input_queue.get_nowait()
            except queue.Empty: pass
        self.input_queue.put(frame)

    def get_result(self):
        try:
            res = self.result_queue.get_nowait()
            self.latest_result = res
            return res
        except queue.Empty:
            return self.latest_result

    def run(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                results = self.detector.detect(frame, imgsz=self.imgsz, device=self.device)
                warnings, roi_color = self.collision_warning.check_collision(results, frame)
                if self.result_queue.full():
                    try: self.result_queue.get_nowait()
                    except queue.Empty: pass
                self.result_queue.put((results, warnings, roi_color))
            except Exception as e:
                print(f"Det Thread Error: {e}")

    def stop(self):
        self.running = False


class LaneThread(threading.Thread):
    def __init__(self, lane_detector):
        super().__init__(daemon=True)
        self.lane_detector = lane_detector
        self.running = True
        self.input_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.latest_result = None

    def detect_async(self, frame):
        if self.input_queue.full():
            try: self.input_queue.get_nowait()
            except queue.Empty: pass
        self.input_queue.put(frame)

    def get_result(self):
        try:
            res = self.result_queue.get_nowait()
            self.latest_result = res
            return res
        except queue.Empty:
            return self.latest_result

    def run(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                result = self.lane_detector.detect(frame)
                if self.result_queue.full():
                    try: self.result_queue.get_nowait()
                    except queue.Empty: pass
                self.result_queue.put(result)
            except Exception as e:
                print(f"Lane Thread Error: {e}")

    def stop(self):
        self.running = False


class PotholeThread(threading.Thread):
    def __init__(self, pothole_detector):
        super().__init__(daemon=True)
        self.pothole_detector = pothole_detector
        self.running = True
        self.input_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.latest_result = []

    def detect_async(self, frame):
        if self.input_queue.full():
            try: self.input_queue.get_nowait()
            except queue.Empty: pass
        self.input_queue.put(frame)

    def get_result(self):
        try:
            res = self.result_queue.get_nowait()
            self.latest_result = res
            return res
        except queue.Empty:
            return self.latest_result

    def run(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                potholes = self.pothole_detector.detect(frame)
                if self.result_queue.full():
                    try: self.result_queue.get_nowait()
                    except queue.Empty: pass
                self.result_queue.put(potholes)
            except Exception as e:
                print(f"Pothole Thread Error: {e}")

    def stop(self):
        self.running = False


class WriterThread(threading.Thread):
    def __init__(self, output_path, width, height, fps=30.0):
        super().__init__(daemon=True)
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.running = True
        self.queue = queue.Queue()

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                writer.write(frame)
            except queue.Empty:
                continue
        writer.release()

    def write(self, frame):
        if not self.running or self.queue.qsize() > 90:
            return
        self.queue.put(frame)

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════════
#  Driving State Machine
# ═══════════════════════════════════════════════════════════════════════════

class DrivingStateMachine:
    DRIVING = "CRUISING"
    BRAKING = "BRAKING"
    REVERSING = "REVERSE"
    AVOID_LEFT = "AVOID LEFT"
    AVOID_RIGHT = "AVOID RIGHT"
    POTHOLE_DODGE = "DODGE POTHOLE"
    STOPPED_RED = "RED LIGHT"
    STOPPED_STOP = "STOP SIGN"

    def __init__(self):
        self.state = self.DRIVING
        self.blocked_since = None
        self.stop_sign_since = None

    def update(self, warnings, obstacle_cx, frame_width, potholes=None):
        now = time.time()
        has_critical = any("CRITICAL" in w for w in warnings)
        has_warning = any("WARNING" in w for w in warnings)
        has_red = "RED LIGHT" in warnings
        has_yellow = "YELLOW LIGHT" in warnings
        has_stop = "STOP SIGN" in warnings
        has_green = "GREEN LIGHT" in warnings

        throttle, steering_offset = 55, 0

        if has_red or (has_yellow and not has_green):
            self.state = self.STOPPED_RED
            self.blocked_since = None
            return self.state, 0, 0

        if has_stop and not has_green:
            if self.stop_sign_since is None:
                self.stop_sign_since = now
            if now - self.stop_sign_since < 2.5:
                self.state = self.STOPPED_STOP
                return self.state, 0, 0
            self.stop_sign_since = None

        if has_critical:
            if self.blocked_since is None:
                self.blocked_since = now
            if now - self.blocked_since > 3.5:
                self.state = self.REVERSING
                return self.state, -25, 0
            self.state = self.BRAKING
            return self.state, 0, 0

        if has_warning:
            self.blocked_since = None
            center = frame_width // 2
            if obstacle_cx > center:
                self.state = self.AVOID_LEFT
                steering_offset = -140
            else:
                self.state = self.AVOID_RIGHT
                steering_offset = 140
            return self.state, 35, steering_offset

        if potholes:
            worst = max(potholes, key=lambda p: {'MINOR': 1, 'MODERATE': 2, 'SEVERE': 3}.get(p['severity'], 0))
            if worst['severity'] in ('MODERATE', 'SEVERE'):
                center = frame_width // 2
                pcx = worst['center'][0]
                self.state = self.POTHOLE_DODGE
                return self.state, 30, -100 if pcx > center else 100

        self.state = self.DRIVING
        self.blocked_since = None
        return self.state, throttle, 0


# ═══════════════════════════════════════════════════════════════════════════
#  Premium Visualization
# ═══════════════════════════════════════════════════════════════════════════

# Color palette (BGR)
C_GREEN     = (72, 225, 100)
C_DARK_GREEN= (40, 140, 55)
C_CYAN      = (230, 220, 50)
C_YELLOW    = (50, 220, 245)
C_ORANGE    = (50, 160, 255)
C_RED       = (60, 60, 240)
C_MAGENTA   = (200, 60, 220)
C_WHITE     = (240, 240, 240)
C_LIGHT_GRAY= (180, 180, 180)
C_DARK_BG   = (30, 30, 30)
C_PANEL_BG  = (20, 20, 20)


def _draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=8):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness == -1:
        # Filled
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def _label_box(img, text, x, y, bg_color, text_color=(255, 255, 255), scale=0.45):
    """Draw a clean label with background pill."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad = 4
    ly = max(y, th + pad * 2 + 2)
    _draw_rounded_rect(img, (x, ly - th - pad * 2), (x + tw + pad * 2, ly), bg_color, -1, 4)
    cv2.putText(img, text, (x + pad, ly - pad),
                cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)


def draw_road_surface(frame, road_result):
    """Draw clean translucent green drivable area."""
    if road_result is None:
        return
    road_mask = road_result.get('road_mask')
    if road_mask is None or not np.any(road_mask > 0):
        return

    # Smooth the mask edges with a large blur
    smooth_mask = cv2.GaussianBlur(road_mask, (21, 21), 0)
    _, smooth_mask = cv2.threshold(smooth_mask, 100, 255, cv2.THRESH_BINARY)

    # Create a clean green overlay
    overlay = frame.copy()
    overlay[smooth_mask > 0] = (
        overlay[smooth_mask > 0].astype(np.int16) +
        np.array([[-30, 50, -30]], dtype=np.int16)
    ).clip(0, 255).astype(np.uint8)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw the road boundary as a clean contour
    contour = road_result.get('road_contour')
    if contour is not None and len(contour) > 5:
        # Smooth the contour
        epsilon = 0.008 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(frame, [approx], -1, C_GREEN, 2, cv2.LINE_AA)

    # Center guide
    info = road_result.get('lanes_info', {})
    lx, rx = info.get('left_x'), info.get('right_x')
    if lx is not None and rx is not None:
        h = frame.shape[0]
        cx = int((lx + rx) / 2)
        # Dashed center line
        for dy in range(int(h * 0.65), h, 20):
            cv2.line(frame, (cx, dy), (cx, min(dy + 10, h)), C_YELLOW, 2, cv2.LINE_AA)

    # Lane lines (if found on marked roads)
    for pt1, pt2 in road_result.get('lane_lines', []):
        cv2.line(frame, pt1, pt2, C_WHITE, 2, cv2.LINE_AA)


def draw_detections(frame, detections, collision_warning, lanes_info, current_warnings):
    """Draw clean, modern bounding boxes."""
    if not detections:
        return
    h, w = frame.shape[:2]

    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x1 >= x2 or y1 >= y2:
            continue

        conf = float(box.conf)
        cls = int(box.cls)
        name = ObjectDetector.get_india_name(cls, detections.names.get(cls, ''))

        # Color by class type
        if cls == 0:
            color = C_MAGENTA   # Person
        elif cls in [15, 16, 17, 18, 19, 20]:
            color = C_CYAN      # Animals
        elif cls == 9:
            color = C_YELLOW    # Traffic light
        elif cls == 11:
            color = C_RED       # Stop sign
        else:
            color = C_GREEN     # Vehicles

        # Distance for vehicles & animals
        dist_text = ""
        if cls in [0, 1, 2, 3, 5, 7, 15, 16, 17, 18, 19, 20]:
            bw = x2 - x1
            dist = collision_warning.estimate_distance(bw, cls)
            dist_text = f" {dist:.1f}m"
            if dist < 8:
                color = C_RED
            elif dist < 20:
                color = C_ORANGE

        # Traffic light override
        if cls == 9:
            if "RED LIGHT" in current_warnings:
                color, name = C_RED, "RED LIGHT"
            elif "YELLOW LIGHT" in current_warnings:
                color, name = C_YELLOW, "YELLOW"
            elif "GREEN LIGHT" in current_warnings:
                color, name = C_GREEN, "GREEN"

        # Draw box — clean corner-only style
        corner_len = min(20, (x2 - x1) // 3, (y2 - y1) // 3)
        t = 2
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, t, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, t, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, t, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, t, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, t, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, t, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, t, cv2.LINE_AA)

        # Label
        label = f"{name} {conf:.0%}{dist_text}"
        _label_box(frame, label, x1, y1, color, C_WHITE if color != C_YELLOW else C_DARK_BG)


def draw_potholes(frame, potholes):
    """Draw pothole detections with clean bounding boxes."""
    for p in potholes:
        x, y, bw, bh = p['bbox']
        conf = p['confidence']
        severity = p['severity']

        color = C_RED if severity == 'SEVERE' else C_ORANGE if severity == 'MODERATE' else C_GREEN

        # Clean rectangle
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2, cv2.LINE_AA)

        # Label
        label = f"pothole {conf:.2f}"
        _label_box(frame, label, x, y, color)


def draw_dashboard(frame, fps, num_det, num_potholes, driving_state,
                   steering_angle, throttle):
    """Draw a polished bottom dashboard panel."""
    h, w = frame.shape[:2]
    panel_h = 52

    # Semi-transparent dark panel at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), C_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Thin accent line at top of panel
    cv2.line(frame, (0, h - panel_h), (w, h - panel_h), C_CYAN, 1, cv2.LINE_AA)

    y_text = h - 18

    # Left section: FPS + counts
    cv2.putText(frame, f"FPS {int(fps)}", (15, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GREEN, 1, cv2.LINE_AA)

    cv2.putText(frame, f"OBJ {num_det}", (100, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1, cv2.LINE_AA)

    if num_potholes > 0:
        cv2.putText(frame, f"POTHOLES {num_potholes}", (175, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_ORANGE, 1, cv2.LINE_AA)

    # Center: Steering visualization
    center_x = w // 2
    bar_w = 100
    cv2.line(frame, (center_x - bar_w, y_text - 5), (center_x + bar_w, y_text - 5),
             C_LIGHT_GRAY, 1, cv2.LINE_AA)
    # Steering indicator dot
    steer_pos = int(center_x + (steering_angle - 90) * (bar_w / 90.0))
    steer_pos = max(center_x - bar_w, min(center_x + bar_w, steer_pos))
    cv2.circle(frame, (steer_pos, y_text - 5), 5, C_YELLOW, -1, cv2.LINE_AA)
    cv2.circle(frame, (center_x, y_text - 5), 2, C_WHITE, -1)

    steer_dir = "LEFT" if steering_angle < 85 else "RIGHT" if steering_angle > 95 else "STRAIGHT"
    cv2.putText(frame, steer_dir, (center_x - 30, y_text + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_LIGHT_GRAY, 1, cv2.LINE_AA)

    # Right section: Throttle bar
    thr_x = w - 130
    thr_w = 80
    thr_fill = max(0, int(thr_w * abs(throttle) / 60))
    cv2.rectangle(frame, (thr_x, y_text - 10), (thr_x + thr_w, y_text), (50, 50, 50), -1)
    thr_color = C_GREEN if throttle > 0 else C_RED if throttle < 0 else C_LIGHT_GRAY
    cv2.rectangle(frame, (thr_x, y_text - 10), (thr_x + thr_fill, y_text), thr_color, -1)
    cv2.rectangle(frame, (thr_x, y_text - 10), (thr_x + thr_w, y_text), C_LIGHT_GRAY, 1)
    thr_label = f"{'R ' if throttle < 0 else ''}{abs(throttle)}"
    cv2.putText(frame, thr_label, (thr_x + thr_w + 5, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHITE, 1, cv2.LINE_AA)


def draw_state_badge(frame, state):
    """Draw the driving state as a clean floating badge at top-center."""
    h, w = frame.shape[:2]

    # Color mapping
    if state == DrivingStateMachine.DRIVING:
        bg, fg = C_DARK_GREEN, C_WHITE
    elif "AVOID" in state or "DODGE" in state:
        bg, fg = C_ORANGE, C_DARK_BG
    else:
        bg, fg = C_RED, C_WHITE

    (tw, th), _ = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    pad = 10
    sx = (w - tw) // 2 - pad
    _draw_rounded_rect(frame, (sx, 10), (sx + tw + pad * 2, 10 + th + pad * 2), bg, -1, 10)
    cv2.putText(frame, state, (sx + pad, 10 + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fg, 2, cv2.LINE_AA)


def draw_warnings(frame, warnings):
    """Draw clean warning alerts."""
    if not warnings:
        return
    h, w = frame.shape[:2]
    has_critical = any("CRITICAL" in wr for wr in warnings)

    # Red border flash for critical
    if has_critical and int(time.time() * 3) % 2 == 0:
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), C_RED, 4)

    # Warnings stacked above dashboard
    y = h - 70
    for warn in reversed(warnings):
        color = C_RED if "CRITICAL" in warn else C_ORANGE
        (tw, th), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = (w - tw) // 2 - 8
        _draw_rounded_rect(frame, (tx, y - th - 8), (tx + tw + 16, y + 4), C_DARK_BG, -1, 6)
        _draw_rounded_rect(frame, (tx, y - th - 8), (tx + tw + 16, y + 4), color, 1, 6)
        cv2.putText(frame, warn, (tx + 8, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y -= th + 16


def draw_collision_roi(frame, collision_warning, roi_color):
    """Draw a subtle ROI zone."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [collision_warning.roi_pixel], roi_color)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    cv2.polylines(frame, [collision_warning.roi_pixel], True, roi_color, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Autonomous Driving System — Indian Roads')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--fix-color', type=str, default=None, choices=['bgr2rgb', 'swap_rg'])
    parser.add_argument('--serial-port', type=str, default='/dev/ttyUSB0')
    parser.add_argument('--no-serial', action='store_true')
    args = parser.parse_args()

    print(f"Starting autonomous driving system (Indian Roads) — source: {args.source}")

    # ── Model ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    pt_paths = [
        '/home/smvec-eee/Desktop/yolo11n.pt',
        os.path.join(project_dir, 'yolov8n.pt'),
        'yolov8n.pt',
    ]
    model_path = None
    for p in pt_paths:
        if os.path.exists(p):
            print(f"Found model: {p}")
            model_path = p
            break
    if model_path is None:
        model_path = 'yolov8n.pt'
        print(f"No local model found, will download: {model_path}")

    try:
        print(f"Loading YOLO model from {model_path}...")
        object_detector = ObjectDetector(model_path=model_path)
    except Exception as e:
        print(f"Warning: Could not load '{model_path}': {e}")
        fallback = os.path.join(project_dir, 'yolov8n.pt')
        if os.path.exists(fallback) and model_path != fallback:
            object_detector = ObjectDetector(model_path=fallback)
        else:
            object_detector = ObjectDetector(model_path='yolov8n.pt')

    lane_detector = LaneDetector()
    pothole_detector = PotholeDetector()

    # ── Camera ──
    source = args.source
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}. Check your camera.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video source: {width}x{height}")

    # ── Writer ──
    writer_thread = None
    if args.output:
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer_thread = WriterThread(args.output, width, height, fps_out)
        writer_thread.start()
        print(f"Recording to: {args.output}")

    # ── Systems ──
    collision_warning = CollisionWarning((height, width))
    serial_controller = None
    if not args.no_serial:
        serial_controller = SerialController(port=args.serial_port)
        try:
            serial_controller.start()
            print(f"Serial started on {args.serial_port}")
        except Exception as e:
            print(f"Serial disabled: {e}")
            serial_controller = None
    else:
        print("Serial Controller disabled (--no-serial)")

    # ── Threads ──
    det_thread = DetectionThread(object_detector, collision_warning, args.imgsz, args.device)
    det_thread.start()
    lane_thread = LaneThread(lane_detector)
    lane_thread.start()
    pothole_thread = PotholeThread(pothole_detector)
    pothole_thread.start()
    print("All threads started. Press 'q' to quit.\n")

    # ── State ──
    sm = DrivingStateMachine()
    prev_time = 0
    frame_count = 0
    detections = None
    current_warnings = []
    roi_color = (0, 255, 0)
    road_result = None
    lanes_info = {}
    potholes = []

    cv2.namedWindow('Autonomous Driving System', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Autonomous Driving System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream.")
                break
            frame_count += 1

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            if args.fix_color == 'bgr2rgb':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif args.fix_color == 'swap_rg':
                frame[:, :, [1, 2]] = frame[:, :, [2, 1]]

            # ── Workers ──
            det_thread.detect_async(frame)
            if frame_count % 2 == 0:
                lane_thread.detect_async(frame)
            if frame_count % 3 == 0:
                pothole_thread.detect_async(frame)

            # ── Collect ──
            det_res = det_thread.get_result()
            if det_res is not None:
                detections, current_warnings, roi_color = det_res

            l_res = lane_thread.get_result()
            if l_res is not None:
                road_result = l_res
                lanes_info = l_res.get('lanes_info', {})

            p_res = pothole_thread.get_result()
            if p_res is not None:
                potholes = p_res

            # ── Obstacle ──
            obstacle_cx = width // 2
            if detections and any("WARNING" in w or "CRITICAL" in w for w in current_warnings):
                closest_d = float('inf')
                for box in detections.boxes:
                    cls = int(box.cls)
                    if cls in collision_warning.obstacle_classes:
                        bx1, _, bx2, by2 = map(int, box.xyxy[0])
                        cx, cy = (bx1 + bx2) // 2, by2
                        if cv2.pointPolygonTest(collision_warning.roi_pixel, (cx, cy), False) >= 0:
                            d = collision_warning.estimate_distance(bx2 - bx1, cls)
                            if d < closest_d:
                                closest_d, obstacle_cx = d, cx

            # ── Decision ──
            state, throttle, steer_off = sm.update(current_warnings, obstacle_cx, width, potholes)

            # ── Steering ──
            lane_cx = width // 2
            lx, rx = lanes_info.get('left_x'), lanes_info.get('right_x')
            if lx is not None and rx is not None:
                lane_cx = int((lx + rx) / 2)
            elif lx is not None:
                lane_cx = int(lx + 150)
            elif rx is not None:
                lane_cx = int(rx - 150)
            lane_cx += steer_off
            steering_angle = max(0, min(180, 90 + int((lane_cx - width // 2) * 0.15)))

            if serial_controller:
                serial_controller.send_command(steering_angle, throttle)

            # ═══ RENDER ═══
            vis = frame.copy()

            # Layer 1: Road surface (green overlay)
            draw_road_surface(vis, road_result)

            # Layer 2: Collision ROI (very subtle)
            draw_collision_roi(vis, collision_warning, roi_color)

            # Layer 3: Object detections (corner-style boxes)
            draw_detections(vis, detections, collision_warning, lanes_info, current_warnings)

            # Layer 4: Potholes
            draw_potholes(vis, potholes)

            # Layer 5: Warnings (above dashboard)
            draw_warnings(vis, current_warnings)

            # Layer 6: State badge (top center)
            draw_state_badge(vis, state)

            # Layer 7: Dashboard panel (bottom bar)
            num_det = len(detections.boxes) if detections else 0
            draw_dashboard(vis, fps, num_det, len(potholes), state,
                          steering_angle, throttle)

            if writer_thread:
                writer_thread.write(vis)

            cv2.imshow('Autonomous Driving System', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        if serial_controller:
            serial_controller.stop()
        for t in [det_thread, lane_thread, pothole_thread]:
            t.stop()
            t.join(timeout=2)
        cap.release()
        if writer_thread:
            writer_thread.stop()
            writer_thread.join(timeout=2)
            print("Output video saved.")
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
