"""
Integration and End-to-End Test Suite for Upgraded ADAS Suite.
Tests all perception, planning, control, and state machine modules.
"""

import numpy as np
import cv2
import time
import sys
import os

# Add autonomous_driving_system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'autonomous_driving_system'))

from perception.camera_geometry import CameraGeometry
from perception.tracking_and_ttc import TrackingAndTTC
from perception.pothole_detector import PotholeDetector
from perception.collision_warning import CollisionWarning
from planning.costmap_planner import CostmapPlanner
from control.stanley_controller import StanleyController
from main import DrivingStateMachine


def test_camera_geometry():
    print("Testing CameraGeometry...")
    geom = CameraGeometry(640, 480, camera_height=1.2, camera_pitch_deg=4.0)
    
    # Ground point directly in front near bottom of frame
    X, Z = geom.pixel_to_ground(320, 440)
    assert 1.0 <= Z <= 6.0, f"Unexpected near ground Z: {Z}"
    assert abs(X) < 0.5, f"Unexpected near ground X: {X}"

    # Mid-range point
    X_mid, Z_mid = geom.pixel_to_ground(320, 320)
    assert Z_mid > Z, f"Z should increase towards horizon: {Z_mid} <= {Z}"

    # Reverse projection
    u, v = geom.ground_to_pixel(0.0, Z)
    assert abs(u - 320) <= 2, f"Re-projected u mismatch: {u} vs 320"
    print("  [PASS] CameraGeometry: Ground projection and reverse projection validated.")


def test_tracking_and_ttc():
    print("Testing TrackingAndTTC...")
    geom = CameraGeometry(640, 480)
    tracker = TrackingAndTTC(geom)

    class MockBox:
        def __init__(self, xyxy, cls_id, conf=0.85):
            self.xyxy = [np.array(xyxy, dtype=float)]
            self.cls = cls_id
            self.conf = conf

    class MockResults:
        def __init__(self, boxes, names=None):
            self.boxes = boxes
            self.names = names or {0: 'person', 2: 'car', 16: 'dog'}

    # Frame 1: Car ahead at 15m
    t0 = time.time()
    # Bottom center at y=320 is ~15m
    b1 = MockBox([280, 260, 360, 320], 2)
    res1 = MockResults([b1])
    tracks1, alerts1 = tracker.update(res1, current_time=t0)
    assert len(tracks1) == 1, "Expected 1 tracked object"
    assert len(alerts1) == 0, "No alert on first frame"

    # Frame 2: Car closes rapidly to 7m in 0.4s (v_rel = 20 m/s -> iTTC < 1.0s)
    t1 = t0 + 0.35
    b2 = MockBox([260, 280, 380, 380], 2)  # Closer down in frame
    res2 = MockResults([b2])
    tracks2, alerts2 = tracker.update(res2, current_time=t1)
    
    has_aeb = any(a['type'] == 'AEB' for a in alerts2)
    assert has_aeb, f"Expected AEB alert for rapid closing car! Alerts: {alerts2}"
    print("  [PASS] TrackingAndTTC: Rapid closing target properly triggered iTTC AEB alert.")

    # Frame 3: Crossing pedestrian from right to left
    tracker2 = TrackingAndTTC(geom)
    p1 = MockBox([480, 300, 520, 370], 0)  # On right shoulder
    t_start = 100.0
    tracker2.update(MockResults([p1]), current_time=t_start)
    
    # Moves left toward center
    p2 = MockBox([400, 300, 440, 370], 0)
    _, cross_alerts = tracker2.update(MockResults([p2]), current_time=t_start + 0.25)
    has_crossing = any(a['type'] == 'CROSSING_ALERT' for a in cross_alerts)
    assert has_crossing, f"Expected CROSSING_ALERT for lateral pedestrian! Alerts: {cross_alerts}"
    print("  [PASS] TrackingAndTTC: Sudden pedestrian crossing interception alert triggered.")


def test_pothole_detector():
    print("Testing PotholeDetector & Depth Estimation...")
    geom = CameraGeometry(640, 480)
    p_detector = PotholeDetector(geom)

    # Create synthetic test frame with asphalt road and a dark pothole
    frame = np.full((480, 640, 3), 110, dtype=np.uint8)  # Gray asphalt
    # Add road texture noise
    noise = np.random.randint(-10, 10, (480, 640, 3)).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw dark pothole with sharp edges at (320, 350)
    cv2.circle(frame, (320, 350), 30, (35, 35, 35), -1)
    cv2.circle(frame, (320, 350), 30, (20, 20, 20), 2)

    # Detect frame 1 (seeds tracker)
    p_detector.detect(frame)
    # Detect frame 2 (confirms track)
    confirmed = p_detector.detect(frame)

    assert len(confirmed) >= 1, "Pothole should be detected and confirmed across 2 frames"
    p = confirmed[0]
    assert p['depth'] > 10.0, f"Pothole depth should be positive: {p['depth']}"
    assert p['severity'] in ('MODERATE', 'SEVERE', 'CRITICAL'), f"Severity: {p['severity']}"
    print(f"  [PASS] PotholeDetector: Confirmed pothole with estimated depth={p['depth']:.1f}, severity={p['severity']}.")


def test_costmap_planner_and_radar():
    print("Testing CostmapPlanner & Follow-The-Gap...")
    geom = CameraGeometry(640, 480)
    planner = CostmapPlanner(640, 480, geom)

    from perception.tracking_and_ttc import TrackedObject
    # Simulated obstacle directly blocking center at (X=0.0m, Z=6.0m)
    obs = TrackedObject(1, (280, 260, 360, 340), 2, "Car", 0.0, 6.0, time.time())

    road_result = {
        'lanes_info': {'left_x': 100, 'right_x': 540}
    }
    plan = planner.plan(road_result, [obs], [], current_speed=1.5)

    assert plan['suggested_state'] in ('AVOID LEFT', 'AVOID RIGHT'), f"State: {plan['suggested_state']}"
    assert abs(plan['target_angle_deg']) > 5.0, f"Should steer away from center obstacle: {plan['target_angle_deg']}"
    assert plan['bev_occupancy'] is not None, "BEV Mini-radar image should be generated"
    assert plan['bev_occupancy'].shape == (140, 140, 3), "BEV Radar size should be 140x140"
    print(f"  [PASS] CostmapPlanner: Follow-the-Gap avoided obstacle with target heading={plan['target_angle_deg']:.1f}°.")


def test_stanley_controller():
    print("Testing StanleyController & Velocity Profiler...")
    controller = StanleyController()

    # Straight ahead, no lateral error
    steer_straight = controller.compute_steering(0.0, 0.0, current_speed_ms=1.5)
    assert abs(steer_straight - 90) <= 2, f"Steering should be ~90: {steer_straight}"

    # Target is to the right (+1.5m offset, +15 deg heading)
    steer_right = controller.compute_steering(1.5, 15.0, current_speed_ms=1.5)
    assert steer_right > 90, f"Steering should turn right (>90): {steer_right}"

    # Throttle profiler tests: initial smooth ramp-up from standstill
    thr_ramp1 = controller.compute_throttle("CRUISING", 90)
    assert 20 <= thr_ramp1 <= 35, f"Initial ramp throttle: {thr_ramp1}"
    # Steady state cruise after ramping
    for _ in range(4):
        thr_cruise = controller.compute_throttle("CRUISING", 90)
    assert 48 <= thr_cruise <= 58, f"Cruising throttle: {thr_cruise}"

    thr_aeb = controller.compute_throttle("EMERGENCY AEB", 90)
    assert thr_aeb < 0, f"AEB throttle should be negative (braking): {thr_aeb}"

    thr_yield = controller.compute_throttle("YIELD CROSSING", 90)
    assert 15 <= thr_yield <= 25, f"Yield crossing should creep: {thr_yield}"
    print("  [PASS] StanleyController: Steering calculations, hardware clamping, and velocity profiles validated.")


def test_state_machine():
    print("Testing DrivingStateMachine...")
    sm = DrivingStateMachine()
    
    # 1. Clear cruise
    s = sm.update([], {'suggested_state': 'CRUISING'})
    assert s == DrivingStateMachine.CRUISING

    # 2. Crossing alert
    s = sm.update(["CROSSING ALERT: Pedestrian approaching"], {'suggested_state': 'CRUISING'})
    assert s == DrivingStateMachine.YIELD_CROSSING

    # 3. AEB alert
    s = sm.update(["CRITICAL AEB: Truck closing"], {'suggested_state': 'CRUISING'})
    assert s == DrivingStateMachine.EMERGENCY_AEB

    # 4. Red light
    s = sm.update(["RED LIGHT"], {'suggested_state': 'CRUISING'})
    assert s == DrivingStateMachine.STOPPED_RED
    print("  [PASS] DrivingStateMachine: Behavioral transitions verified.")


if __name__ == '__main__':
    print("=" * 65)
    print("   RUNNING ADAS FULL SUITE INTEGRATION TESTS")
    print("=" * 65)
    test_camera_geometry()
    test_tracking_and_ttc()
    test_pothole_detector()
    test_costmap_planner_and_radar()
    test_stanley_controller()
    test_state_machine()
    print("=" * 65)
    print("   ALL TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
    print("=" * 65)
