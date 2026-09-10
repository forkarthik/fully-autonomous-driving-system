"""
Autonomous Driving System (ADAS) for NVIDIA Jetson Orin Nano (JetPack 6 / L4T R36+)
Next-Gen Cyberpunk / Tesla FSD / Swaayatt Robots Edition:
  - 3D Holographic Perspective Trajectory Ribbon with Animated Flowing Chevrons
  - Multi-Hypothesis Motion Primitive Rollouts (Swaayatt Robots Style)
  - 3D Ground Footprints with Dynamic Threat Concentric Pulsing
  - Topographical Pothole Sonar Rings & Depth Isoline Badges
  - Glassmorphism Cockpit Dashboard with Digital Speedometer & Steering Compass
  - High-Tech 360° Circular Tactical BEV Radar
  - Follow-The-Gap Free Space Planner & Stanley Lateral Control Law
  - Instantaneous Time-To-Collision (iTTC) & Automatic Emergency Braking (AEB)
  - Audible Piezo Buzzer Warning Triggers for ESP32
"""

import cv2
import argparse
import numpy as np
import threading
import queue
import time
import os
import math

from perception.object_detector import ObjectDetector
from perception.lane_detector import LaneDetector
from perception.collision_warning import CollisionWarning
from perception.pothole_detector import PotholeDetector
from perception.camera_geometry import CameraGeometry
from perception.tracking_and_ttc import TrackingAndTTC
from planning.costmap_planner import CostmapPlanner
from control.stanley_controller import StanleyController
from utils.serial_controller import SerialController


# ═══════════════════════════════════════════════════════════════════════════
#  Worker Threads (Multi-threaded asynchronous perception for Jetson GPU)
# ═══════════════════════════════════════════════════════════════════════════

class DetectionThread(threading.Thread):
    def __init__(self, detector, collision_warning, tracking_and_ttc, imgsz, device):
        super().__init__(daemon=True)
        self.detector = detector
        self.collision_warning = collision_warning
        self.tracking_and_ttc = tracking_and_ttc
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
                now_ts = time.time()
                results = self.detector.detect(frame, imgsz=self.imgsz, device=self.device)
                active_tracks, safety_alerts = self.tracking_and_ttc.update(results, now_ts)
                warnings, roi_color = self.collision_warning.check_collision(results, frame, safety_alerts)

                if self.result_queue.full():
                    try: self.result_queue.get_nowait()
                    except queue.Empty: pass
                self.result_queue.put((results, active_tracks, safety_alerts, warnings, roi_color))
            except Exception as e:
                print(f"Detection Thread Error: {e}")

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
#  Tier4 Behavioral State Machine
# ═══════════════════════════════════════════════════════════════════════════

class DrivingStateMachine:
    CRUISING = "CRUISING"
    EMERGENCY_AEB = "EMERGENCY AEB"
    BRAKING = "BRAKING"
    REVERSING = "REVERSE"
    YIELD_CROSSING = "YIELD CROSSING"
    AVOID_LEFT = "AVOID LEFT"
    AVOID_RIGHT = "AVOID RIGHT"
    POTHOLE_DODGE = "DODGE POTHOLE"
    STOPPED_RED = "RED LIGHT"
    STOPPED_STOP = "STOP SIGN"
    SLOW = "SLOW"

    def __init__(self):
        self.state = self.CRUISING
        self.blocked_since = None
        self.stop_sign_since = None

    def update(self, warnings, plan_result, potholes=None):
        now = time.time()
        has_aeb = any("AEB" in w for w in warnings)
        has_critical = any("CRITICAL" in w for w in warnings)
        has_crossing = any("CROSSING" in w for w in warnings)
        has_red = any("RED LIGHT" in w for w in warnings)
        has_yellow = any("YELLOW LIGHT" in w for w in warnings)
        has_green = any("GREEN LIGHT" in w for w in warnings)
        has_stop = any("STOP SIGN" in w for w in warnings)

        planner_state = plan_result.get('suggested_state', 'CRUISING') if plan_result else 'CRUISING'

        # Traffic signals
        if has_red or (has_yellow and not has_green):
            self.state = self.STOPPED_RED
            self.blocked_since = None
            return self.state

        if has_stop and not has_green:
            if self.stop_sign_since is None:
                self.stop_sign_since = now
            if now - self.stop_sign_since < 2.5:
                self.state = self.STOPPED_STOP
                return self.state
            self.stop_sign_since = None

        # Instantaneous iTTC AEB
        if has_aeb:
            self.state = self.EMERGENCY_AEB
            if self.blocked_since is None:
                self.blocked_since = now
            if now - self.blocked_since > 3.0:
                self.state = self.REVERSING
            return self.state

        # Critical proximity
        if has_critical:
            if self.blocked_since is None:
                self.blocked_since = now
            if now - self.blocked_since > 3.5:
                self.state = self.REVERSING
            else:
                self.state = self.BRAKING
            return self.state

        self.blocked_since = None

        # Anticipatory Crossing Interception
        if has_crossing:
            self.state = self.YIELD_CROSSING
            return self.state

        # Pothole Dodge
        if planner_state == "DODGE POTHOLE" or (potholes and any(p.get('severity') in ('SEVERE', 'CRITICAL') for p in potholes)):
            self.state = self.POTHOLE_DODGE
            return self.state

        # Corridor Navigation
        if planner_state in ("AVOID LEFT", "AVOID RIGHT"):
            self.state = planner_state
            return self.state

        if planner_state == "SLOW":
            self.state = self.SLOW
            return self.state

        self.state = self.CRUISING
        return self.state


# ═══════════════════════════════════════════════════════════════════════════
#  Next-Gen Cyberpunk / Holographic HUD Palette & Utilities
# ═══════════════════════════════════════════════════════════════════════════

# Tailored HSL-tuned BGR Colors for sleek aesthetic
C_NEON_GREEN  = (100, 255, 120)
C_DARK_GREEN  = (30, 110, 45)
C_CYAN_GLOW   = (255, 235, 60)
C_ELECTRIC_BLU= (255, 190, 40)
C_NEON_AMBER  = (40, 200, 255)
C_CRITICAL_RED= (60, 60, 245)
C_HOT_MAGENTA = (220, 70, 220)
C_TEXT_WHITE  = (245, 245, 250)
C_MUTED_GRAY  = (160, 165, 175)
C_DARK_DECK   = (14, 15, 18)
C_CARD_BG     = (22, 24, 28)
C_ACCENT_LINE = (230, 210, 50)


def _draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=6):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)


def _label_box(img, text, x, y, bg_color, text_color=(255, 255, 255), scale=0.45):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad = 4
    ly = max(y, th + pad * 2 + 2)
    _draw_rounded_rect(img, (x, ly - th - pad * 2), (x + tw + pad * 2, ly), bg_color, -1, 4)
    cv2.putText(img, text, (x + pad, ly - pad),
                cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 1: 3D Holographic Perspective Trajectory Ribbon & Motion Primitives
# ═══════════════════════════════════════════════════════════════════════════

def draw_3d_trajectory_ribbon(frame, road_result, plan_result, camera_geometry, steering_angle):
    """
    Renders a glowing 3D perspective trajectory ribbon curving with steering angle
    and displays Swaayatt Robots multi-hypothesis motion primitive candidate rays.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1. Translucent Road Surface Mask
    if road_result is not None:
        road_mask = road_result.get('road_mask')
        if road_mask is not None and np.any(road_mask > 0):
            smooth_mask = cv2.GaussianBlur(road_mask, (19, 19), 0)
            _, smooth_mask = cv2.threshold(smooth_mask, 80, 255, cv2.THRESH_BINARY)
            overlay[smooth_mask > 0] = (
                overlay[smooth_mask > 0].astype(np.int16) +
                np.array([[-25, 45, -25]], dtype=np.int16)
            ).clip(0, 255).astype(np.uint8)

            # Lateral boundary edge glow
            contour = road_result.get('road_contour')
            if contour is not None and len(contour) > 5:
                cv2.drawContours(frame, [contour], -1, C_NEON_GREEN, 1, cv2.LINE_AA)

    # 2. Swaayatt-Style Multi-Hypothesis Motion Primitive Rollouts
    # 5 Candidate rays: Hard-Left (-24°), Soft-Left (-12°), Center (0°), Soft-Right (+12°), Hard-Right (+24°)
    horizon_y = int(h * 0.44)
    start_pt = (w // 2, h - 55)
    aim_x = plan_result.get('target_x_px', w // 2) if plan_result else w // 2

    candidate_offsets = [-70, -35, 0, 35, 70]
    for off in candidate_offsets:
        cand_x = (w // 2) + off
        cv2.line(frame, start_pt, (cand_x, horizon_y + 15), (60, 75, 85), 1, cv2.LINE_AA)

    # 3. Dynamic Curving 3D Perspective Ribbon (Tesla FSD / Swaayatt Style)
    # Curvature derived from steering angle (90 is straight, <90 left, >90 right)
    steer_delta_rad = np.radians(steering_angle - 90.0)
    steer_curvature = np.tan(steer_delta_rad) * 0.04

    left_pts, right_pts = [], []
    half_w_m = 0.9  # 1.8m vehicle lane corridor
    z_steps = np.linspace(1.8, 22.0, 18)

    for z in z_steps:
        # Lateral displacement along curvature: x = 0.5 * curvature * z^2
        lateral_x = 0.5 * steer_curvature * (z ** 2)
        xl = lateral_x - half_w_m
        xr = lateral_x + half_w_m

        ul, vl = camera_geometry.ground_to_pixel(xl, z)
        ur, vr = camera_geometry.ground_to_pixel(xr, z)

        if vl > horizon_y and vr > horizon_y:
            left_pts.append((ul, vl))
            right_pts.append((ur, vr))

    if len(left_pts) >= 4:
        ribbon_poly = np.array(left_pts + right_pts[::-1], dtype=np.int32)
        # Ribbon glow color based on planner state
        is_evading = plan_result and plan_result.get('suggested_state') != 'CRUISING'
        ribbon_color = (30, 160, 240) if is_evading else (45, 140, 65)
        cv2.fillPoly(overlay, [ribbon_poly], ribbon_color)

        # Draw ribbon borders
        cv2.polylines(frame, [np.array(left_pts, dtype=np.int32)], False, C_CYAN_GLOW, 2, cv2.LINE_AA)
        cv2.polylines(frame, [np.array(right_pts, dtype=np.int32)], False, C_CYAN_GLOW, 2, cv2.LINE_AA)

        # Flowing forward chevron arrows (pulsing along time)
        pulse_offset = int((time.time() * 6.0) % len(left_pts))
        for idx in range(pulse_offset, len(left_pts), 4):
            if idx < len(left_pts):
                lx, ly = left_pts[idx]
                rx, ry = right_pts[idx]
                cx, cy = (lx + rx) // 2, (ly + ry) // 2
                tip_y = cy - 8
                cv2.line(frame, (cx - 12, cy + 4), (cx, tip_y), C_NEON_AMBER, 2, cv2.LINE_AA)
                cv2.line(frame, (cx + 12, cy + 4), (cx, tip_y), C_NEON_AMBER, 2, cv2.LINE_AA)

    # Blend translucent overlays
    cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

    # 4. Target Horizon Lock-on Reticle
    cv2.circle(frame, (aim_x, horizon_y), 5, C_NEON_AMBER, -1, cv2.LINE_AA)
    cv2.circle(frame, (aim_x, horizon_y), 10, C_TEXT_WHITE, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 2 & 3: 3D Ground Footprints, Holographic Boxes & iTTC Lock-On
# ═══════════════════════════════════════════════════════════════════════════

def draw_nextgen_detections(frame, detections, active_tracks, camera_geometry, current_warnings):
    """
    Draws 3D perspective ground footprint ellipses, holographic bracketed boxes,
    live motion vector arrows, and tactical target lock-on crosshairs.
    """
    if not detections or detections.boxes is None:
        return
    h, w = frame.shape[:2]

    # Find highest threat target (lowest iTTC or closest distance in path)
    highest_threat_trk = None
    min_threat_metric = float('inf')

    for trk in active_tracks or []:
        if abs(trk.X) < 1.4:
            metric = trk.ittc if trk.is_closing else trk.Z
            if metric < min_threat_metric:
                min_threat_metric = metric
                highest_threat_trk = trk

    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x1 >= x2 or y1 >= y2:
            continue

        cls_id = int(box.cls)
        name = ObjectDetector.get_india_name(cls_id, detections.names.get(cls_id, ''))
        conf = float(box.conf)

        # Find matching track
        matched_trk = None
        bx_center = (x1 + x2) / 2.0
        for trk in active_tracks or []:
            tx1, ty1, tx2, ty2 = trk.bbox
            tcx = (tx1 + tx2) / 2.0
            if abs(tcx - bx_center) < 35 and abs(ty2 - y2) < 35:
                matched_trk = trk
                break

        # Theme color
        if cls_id == 0:
            base_color = C_HOT_MAGENTA  # Pedestrians
        elif cls_id in (15, 16, 17, 18, 19, 20):
            base_color = C_CYAN_GLOW    # Cattle, Dogs, Animals
        elif cls_id == 9:
            base_color = C_NEON_AMBER   # Traffic Light
        elif cls_id == 11:
            base_color = C_CRITICAL_RED # Stop Sign
        else:
            base_color = C_NEON_GREEN   # Vehicles / Autos / Trucks

        dist_m = matched_trk.Z if matched_trk else 10.0
        speed_tag = ""
        ittc_tag = ""

        if matched_trk:
            if matched_trk.is_closing and matched_trk.vz > 0.35:
                speed_tag = f" -{matched_trk.vz:.1f}m/s"
                if matched_trk.ittc < 2.5:
                    ittc_tag = f" [iTTC: {matched_trk.ittc:.1f}s]"
                    base_color = C_CRITICAL_RED
            elif matched_trk.is_crossing:
                speed_tag = f" >{abs(matched_trk.vx):.1f}m/s"
                base_color = C_NEON_AMBER
            elif matched_trk.Z < 6.0:
                base_color = C_CRITICAL_RED

        # 1. 3D Perspective Ground Footprint Ellipse on the asphalt
        foot_w = max(16, (x2 - x1) // 2)
        foot_h = max(6, int(foot_w * 0.35))
        foot_center = (int(bx_center), y2 - 2)

        # Concentric pulsing ring if closing hazard
        if matched_trk and matched_trk.is_closing and matched_trk.ittc < 2.5:
            pulse_rad = int(foot_w * (1.0 + 0.3 * np.sin(time.time() * 10.0)))
            cv2.ellipse(frame, foot_center, (pulse_rad, int(pulse_rad * 0.35)), 0, 0, 360, C_CRITICAL_RED, 1, cv2.LINE_AA)

        cv2.ellipse(frame, foot_center, (foot_w, foot_h), 0, 0, 360, base_color, 2, cv2.LINE_AA)

        # 2. Motion Velocity Vector Arrow
        if matched_trk and (abs(matched_trk.vx) > 0.4 or abs(matched_trk.vz) > 0.4):
            v_dx = int(matched_trk.vx * 18.0)
            v_dy = int(-matched_trk.vz * 12.0)
            arr_end = (foot_center[0] + v_dx, foot_center[1] + v_dy)
            cv2.arrowedLine(frame, foot_center, arr_end, C_TEXT_WHITE, 2, cv2.LINE_AA, tipLength=0.35)

        # 3. Holographic Bracketed Box
        bw = x2 - x1
        bh = y2 - y1
        b_len = max(8, min(22, bw // 4, bh // 4))

        # Glowing corner brackets
        cv2.line(frame, (x1, y1), (x1 + b_len, y1), base_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + b_len), base_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2 - b_len, y1), base_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + b_len), base_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1 + b_len, y2), base_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - b_len), base_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - b_len, y2), base_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - b_len), base_color, 2, cv2.LINE_AA)

        # 4. Lock-on Crosshair if this is the Highest Threat Target
        if matched_trk and matched_trk == highest_threat_trk and matched_trk.Z < 18.0:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 6, C_CRITICAL_RED, 1, cv2.LINE_AA)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), C_CRITICAL_RED, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), C_CRITICAL_RED, 1, cv2.LINE_AA)

        # 5. Holographic Pill Tag
        label_text = f"{name} {dist_m:.1f}m{speed_tag}{ittc_tag}"
        _label_box(frame, label_text, x1, y1, base_color, C_DARK_DECK if base_color in (C_CYAN_GLOW, C_NEON_AMBER) else C_TEXT_WHITE)


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 4: Topographical Pothole Sonar Rings & Depth Badges
# ═══════════════════════════════════════════════════════════════════════════

def draw_topographical_potholes(frame, potholes):
    """
    Renders pulsing sonar warning rings and topographical depth badges
    inspired by Swaayatt Robots road anomaly contour maps.
    """
    for p in potholes:
        x, y, bw, bh = p['bbox']
        cx, cy = p['center']
        severity = p.get('severity', 'MINOR')
        depth = p.get('depth', 0.0)

        # Color coding
        p_color = C_CRITICAL_RED if severity in ('CRITICAL', 'SEVERE') else C_NEON_AMBER if severity == 'MODERATE' else C_NEON_GREEN

        # 1. Concentric Sonar Ripples on the asphalt
        sonar_radius = int((bw // 2) * (1.0 + 0.35 * np.sin(time.time() * 8.0)))
        cv2.ellipse(frame, (cx, cy), (sonar_radius, int(sonar_radius * 0.45)), 0, 0, 360, p_color, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (bw // 2, max(4, int(bh // 2))), 0, 0, 360, p_color, 2, cv2.LINE_AA)

        # 2. Depth Badge
        depth_cm = depth * 0.25  # Approximate centimeters
        badge_text = f"POTHOLE -{depth_cm:.1f}cm [{severity}]"
        _label_box(frame, badge_text, x, y, p_color, C_DARK_DECK if p_color == C_NEON_AMBER else C_TEXT_WHITE)


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 7: Glassmorphism Cockpit Dashboard & Steering Compass
# ═══════════════════════════════════════════════════════════════════════════

def draw_cockpit_dashboard(frame, fps, num_det, num_potholes, driving_state, steering_angle, throttle):
    """
    Futuristic Tesla FSD & F1 style bottom cockpit dashboard:
      - Digital speedometer with KM/H readout
      - Curved steering compass dial
      - Throttle & regenerative brake power meter
      - Jetson MAXN hardware telemetry
    """
    h, w = frame.shape[:2]
    deck_h = 56

    # Glassmorphism dark panel at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - deck_h), (w, h), C_DARK_DECK, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Glowing top cyan accent line
    cv2.line(frame, (0, h - deck_h), (w, h - deck_h), C_ACCENT_LINE, 1, cv2.LINE_AA)

    # 1. Speedometer (Left Deck)
    # Estimate vehicle speed in KM/H from throttle
    speed_kmh = max(0, int(abs(throttle) * 0.45))
    cv2.putText(frame, f"{speed_kmh:02d}", (22, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, C_TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "KM/H", (65, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_MUTED_GRAY, 1, cv2.LINE_AA)

    # Hardware & Objects Mini-tags
    cv2.putText(frame, f"FPS {int(fps)}", (115, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_NEON_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"OBJ {num_det} | POTH {num_potholes}", (115, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN_GLOW, 1, cv2.LINE_AA)

    # 2. Steering Compass Dial (Center Deck)
    center_x = w // 2
    compass_y = h - 26
    bar_span = 85

    # Center zero tick
    cv2.line(frame, (center_x, compass_y - 12), (center_x, compass_y + 12), C_MUTED_GRAY, 1, cv2.LINE_AA)
    # Range rail
    cv2.line(frame, (center_x - bar_span, compass_y), (center_x + bar_span, compass_y), (50, 55, 65), 1, cv2.LINE_AA)

    # Ticks along steering rail
    for tick_deg in [-60, -30, 30, 60]:
        tx = int(center_x + (tick_deg / 90.0) * bar_span)
        cv2.line(frame, (tx, compass_y - 4), (tx, compass_y + 4), (70, 75, 85), 1, cv2.LINE_AA)

    # Dynamic needle indicator
    steer_offset = (steering_angle - 90.0)
    needle_x = int(center_x + (steer_offset / 75.0) * bar_span)
    needle_x = max(center_x - bar_span, min(center_x + bar_span, needle_x))

    needle_color = C_NEON_AMBER if abs(steer_offset) > 10 else C_NEON_GREEN
    cv2.circle(frame, (needle_x, compass_y), 5, needle_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (needle_x, compass_y), 8, C_TEXT_WHITE, 1, cv2.LINE_AA)

    steer_text = f"{steering_angle}°"
    if steering_angle < 82:
        steer_text += " LEFT"
    elif steering_angle > 98:
        steer_text += " RIGHT"
    else:
        steer_text += " CENTER"

    cv2.putText(frame, steer_text, (center_x - 30, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_MUTED_GRAY, 1, cv2.LINE_AA)

    # 3. Throttle & Regenerative Brake Meter (Right Deck)
    thr_x = w - 135
    thr_w = 70
    thr_y = h - 30

    cv2.rectangle(frame, (thr_x, thr_y), (thr_x + thr_w, thr_y + 12), (35, 38, 44), -1)
    thr_fill = int(thr_w * min(1.0, abs(throttle) / 60.0))
    bar_color = C_NEON_GREEN if throttle > 0 else C_CRITICAL_RED if throttle < 0 else C_MUTED_GRAY
    cv2.rectangle(frame, (thr_x, thr_y), (thr_x + thr_fill, thr_y + 12), bar_color, -1)
    cv2.rectangle(frame, (thr_x, thr_y), (thr_x + thr_w, thr_y + 12), (60, 65, 75), 1)

    thr_tag = f"{'REVERSE ' if throttle < 0 else 'PWR '}{abs(throttle)}%"
    cv2.putText(frame, thr_tag, (thr_x, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT_WHITE, 1, cv2.LINE_AA)

    # Jetson Hardware Mode Badge
    cv2.putText(frame, "JETSON MAXN 15W", (w - 135, h - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 210, 100), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 6: Floating Holographic State Badge & Alert Stack
# ═══════════════════════════════════════════════════════════════════════════

def draw_state_and_alerts(frame, state, warnings):
    """Renders top-center floating state badge and dynamic threat alert banner."""
    h, w = frame.shape[:2]

    # 1. State Badge (Top Center)
    if state == DrivingStateMachine.CRUISING:
        bg_col, fg_col = C_DARK_GREEN, C_TEXT_WHITE
    elif "AVOID" in state or "DODGE" in state or "YIELD" in state:
        bg_col, fg_col = C_NEON_AMBER, C_DARK_DECK
    elif state == "SLOW":
        bg_col, fg_col = (40, 180, 230), C_DARK_DECK
    else:
        bg_col, fg_col = C_CRITICAL_RED, C_TEXT_WHITE

    state_label = f"ADAS: {state}"
    (tw, th), _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
    pad = 12
    sx = (w - tw) // 2 - pad
    _draw_rounded_rect(frame, (sx, 8), (sx + tw + pad * 2, 8 + th + pad * 2), bg_col, -1, 8)
    _draw_rounded_rect(frame, (sx, 8), (sx + tw + pad * 2, 8 + th + pad * 2), C_TEXT_WHITE, 1, 8)
    cv2.putText(frame, state_label, (sx + pad, 8 + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, fg_col, 2, cv2.LINE_AA)

    # 2. Warnings Alert Stack (Above Dashboard)
    if not warnings:
        return

    has_critical = any("CRITICAL" in wr or "AEB" in wr for wr in warnings)
    if has_critical and int(time.time() * 4) % 2 == 0:
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), C_CRITICAL_RED, 4)

    y = h - 74
    for warn in reversed(warnings[:3]):
        is_crit = ("CRITICAL" in warn or "AEB" in warn)
        color = C_CRITICAL_RED if is_crit else C_NEON_AMBER
        (wtw, wth), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        tx = (w - wtw) // 2 - 10
        _draw_rounded_rect(frame, (tx, y - wth - 6), (tx + wtw + 20, y + 5), C_CARD_BG, -1, 5)
        _draw_rounded_rect(frame, (tx, y - wth - 6), (tx + wtw + 20, y + 5), color, 1, 5)
        cv2.putText(frame, warn, (tx + 10, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        y -= wth + 16


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 8: Tactical Circular BEV Radar Overlay
# ═══════════════════════════════════════════════════════════════════════════

def draw_tactical_bev_radar(frame, plan_result):
    """Renders sleek 145x145 tactical radar in upper-right corner."""
    if not plan_result or plan_result.get('bev_occupancy') is None:
        return
    bev_img = plan_result['bev_occupancy']
    rh, rw = bev_img.shape[:2]
    margin = 8
    x1 = frame.shape[1] - rw - margin
    y1 = margin

    # Subtly rounded HUD frame
    frame[y1:y1 + rh, x1:x1 + rw] = bev_img
    cv2.rectangle(frame, (x1, y1), (x1 + rw, y1 + rh), C_ACCENT_LINE, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Main System Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Next-Gen ADAS Suite for NVIDIA Jetson Orin Nano')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--fix-color', type=str, default=None, choices=['bgr2rgb', 'swap_rg'])
    parser.add_argument('--serial-port', type=str, default='/dev/ttyUSB0')
    parser.add_argument('--no-serial', action='store_true')
    args = parser.parse_args()

    print("=" * 65)
    print("  🚀 FULLY AUTONOMOUS DRIVING SYSTEM (ADAS) — JETSON ORIN NANO")
    print("=" * 65)
    print(f"  Source Video/Camera  : {args.source}")
    print(f"  Target Resolution    : 640x480 @ 30 FPS (Jetson Sweet Spot)")
    print(f"  Motor Control        : {'Disabled (--no-serial)' if args.no_serial else args.serial_port}")
    print("=" * 65 + "\n")

    # ── Model Selection (TensorRT .engine preferred, .pt fallback) ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    engine_paths = [
        os.path.join(project_dir, 'yolo11n.engine'),
        '/home/smvec-eee/Desktop/yolo11n.engine',
        'yolo11n.engine',
    ]
    pt_paths = [
        os.path.join(project_dir, 'yolov8n.pt'),
        'yolov8n.pt',
        '/home/smvec-eee/Desktop/yolo11n.pt',
    ]
    model_path = None
    for p in engine_paths + pt_paths:
        if os.path.exists(p):
            print(f"[INFO] Found accelerated model: {p}")
            model_path = p
            break
    if model_path is None:
        model_path = 'yolov8n.pt'
        print(f"[INFO] Downloading default weights: {model_path}")

    object_detector = ObjectDetector(model_path=model_path)

    # ── Camera Setup (640x480 for 30 FPS) ──
    source = args.source
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {source}. Check connections.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream initialized: {width}x{height} @ 30 FPS")

    # ── Perception, Geometry, Planning & Control Modules ──
    camera_geometry = CameraGeometry(frame_width=width, frame_height=height)
    tracking_and_ttc = TrackingAndTTC(camera_geometry)
    lane_detector = LaneDetector()
    pothole_detector = PotholeDetector(camera_geometry)
    collision_warning = CollisionWarning((height, width), camera_geometry)
    costmap_planner = CostmapPlanner(width, height, camera_geometry)
    stanley_controller = StanleyController()

    # ── ESP32 Serial Communication ──
    serial_controller = None
    if not args.no_serial:
        serial_controller = SerialController(port=args.serial_port)
        try:
            serial_controller.start()
            print(f"[INFO] Serial linked to ESP32 on {args.serial_port}")
        except Exception as e:
            print(f"[WARN] Serial disabled: {e}")
            serial_controller = None

    # ── Video Recording Thread ──
    writer_thread = None
    if args.output:
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer_thread = WriterThread(args.output, width, height, fps_out)
        writer_thread.start()
        print(f"[INFO] Recording session to: {args.output}")

    # ── Asynchronous AI Worker Threads ──
    det_thread = DetectionThread(object_detector, collision_warning, tracking_and_ttc, args.imgsz, args.device)
    det_thread.start()
    lane_thread = LaneThread(lane_detector)
    lane_thread.start()
    pothole_thread = PotholeThread(pothole_detector)
    pothole_thread.start()
    print("[INFO] Asynchronous perception pipeline running. Press 'q' to exit.\n")

    # ── Runtime State ──
    sm = DrivingStateMachine()
    prev_time = 0
    frame_count = 0
    detections = None
    active_tracks = []
    safety_alerts = []
    current_warnings = []
    roi_color = (0, 255, 0)
    road_result = None
    potholes = []
    plan_result = None
    steering_angle = 90
    throttle = 0

    cv2.namedWindow('Autonomous Driving System', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Autonomous Driving System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video feed ended.")
                break
            frame_count += 1

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            # Color fix for Linux USB camera driver
            if args.fix_color == 'bgr2rgb':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif args.fix_color == 'swap_rg':
                frame[:, :, [1, 2]] = frame[:, :, [2, 1]]

            # ── Async Dispatch ──
            det_thread.detect_async(frame)
            if frame_count % 2 == 0:
                lane_thread.detect_async(frame)
            if frame_count % 3 == 0:
                pothole_thread.detect_async(frame)

            # ── Collect Latest Outputs ──
            det_res = det_thread.get_result()
            if det_res is not None:
                detections, active_tracks, safety_alerts, current_warnings, roi_color = det_res

            l_res = lane_thread.get_result()
            if l_res is not None:
                road_result = l_res

            p_res = pothole_thread.get_result()
            if p_res is not None:
                potholes = p_res

            # ── Follow-the-Gap Corridor Planning ──
            plan_result = costmap_planner.plan(
                road_result,
                active_tracks,
                potholes,
                current_speed=max(0.5, abs(throttle) * 0.035)
            )

            # ── State Machine Evaluation ──
            state = sm.update(current_warnings, plan_result, potholes)

            # ── Stanley Control Law & Velocity Profiling ──
            target_lat = plan_result.get('target_lateral_offset_m', 0.0)
            target_heading = plan_result.get('target_angle_deg', 0.0)

            if state in ("RED LIGHT", "STOP SIGN", "EMERGENCY AEB", "BRAKING"):
                target_lat = 0.0
                target_heading = 0.0

            steering_angle = stanley_controller.compute_steering(
                target_lat,
                target_heading,
                current_speed_ms=max(0.6, abs(throttle) * 0.035)
            )
            throttle = stanley_controller.compute_throttle(state, steering_angle)

            # Send physical PWM signals to ESP32
            if serial_controller:
                alarm_level = 2 if "AEB" in state else (1 if ("CROSSING" in state or "WARN" in state or "ALERT" in state or "BRAKING" in state) else 0)
                serial_controller.send_command(steering_angle, throttle, alarm=alarm_level)

            # ═══ RENDER NEXT-GEN CYBERPUNK HUD ═══
            vis = frame.copy()

            # Layer 1: 3D Holographic Perspective Trajectory Ribbon & Motion Primitives
            draw_3d_trajectory_ribbon(vis, road_result, plan_result, camera_geometry, steering_angle)

            # Layer 2 & 3: 3D Ground Footprints, Holographic Bounding Boxes & iTTC Lock-On
            draw_nextgen_detections(vis, detections, active_tracks, camera_geometry, current_warnings)

            # Layer 4: Topographical Pothole Sonar Rings & Elevation Badges
            draw_topographical_potholes(vis, potholes)

            # Layer 5 & 6: Floating State Pill & Dynamic Threat Alert Stack
            draw_state_and_alerts(vis, state, current_warnings)

            # Layer 7: Glassmorphism Cockpit Dashboard & Steering Compass
            num_det = len(detections.boxes) if detections and detections.boxes is not None else 0
            draw_cockpit_dashboard(vis, fps, num_det, len(potholes), state, steering_angle, throttle)

            # Layer 8: Tactical Circular BEV Radar Overlay
            draw_tactical_bev_radar(vis, plan_result)

            if writer_thread:
                writer_thread.write(vis)

            cv2.imshow('Autonomous Driving System', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] Shutting down ADAS cleanly...")
        if serial_controller:
            serial_controller.stop()
        for t in [det_thread, lane_thread, pothole_thread]:
            t.stop()
            t.join(timeout=2)
        cap.release()
        if writer_thread:
            writer_thread.stop()
            writer_thread.join(timeout=2)
            print("[INFO] Video artifact saved.")
        cv2.destroyAllWindows()
        print("[INFO] System shutdown complete.")


if __name__ == "__main__":
    main()
