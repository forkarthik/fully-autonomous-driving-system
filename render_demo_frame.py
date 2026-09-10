"""
Generates a demonstration frame showcasing the Next-Gen Cyberpunk / Swaayatt Robots HUD
and saves it to test_hud_output.jpg for visual verification.
"""

import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'autonomous_driving_system'))

from perception.camera_geometry import CameraGeometry
from perception.tracking_and_ttc import TrackingAndTTC, TrackedObject
from perception.pothole_detector import PotholeDetector
from perception.lane_detector import LaneDetector
from perception.collision_warning import CollisionWarning
from planning.costmap_planner import CostmapPlanner
from control.stanley_controller import StanleyController
from main import (
    draw_3d_trajectory_ribbon,
    draw_nextgen_detections,
    draw_topographical_potholes,
    draw_state_and_alerts,
    draw_cockpit_dashboard,
    draw_tactical_bev_radar
)

def main():
    img_path = 'test.jpg'
    if os.path.exists(img_path):
        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, (640, 480))
    else:
        frame = np.full((480, 640, 3), 70, dtype=np.uint8)

    w, h = 640, 480
    camera_geometry = CameraGeometry(frame_width=w, frame_height=h)
    lane_detector = LaneDetector()
    pothole_detector = PotholeDetector(camera_geometry)
    costmap_planner = CostmapPlanner(w, h, camera_geometry)
    stanley_controller = StanleyController()

    # Detect drivable road
    road_result = lane_detector.detect(frame)

    # Simulated tracked objects on Indian road
    now = time.time()
    # 1. Approaching Auto-rickshaw ahead at 8.5m
    trk1 = TrackedObject(1, (270, 240, 370, 330), 2, "Auto-rickshaw", 0.1, 8.5, now)
    trk1.vz = 2.4
    trk1.is_closing = True
    trk1.ittc = 3.5

    # 2. Crossing pedestrian from right shoulder at 11.2m
    trk2 = TrackedObject(2, (460, 260, 500, 340), 0, "Person", 1.8, 11.2, now)
    trk2.vx = -1.2
    trk2.is_crossing = True
    trk2.crossing_time = 1.4

    # 3. Stray Cow on left edge at 14.0m
    trk3 = TrackedObject(3, (120, 250, 210, 320), 19, "Cow / Bull", -2.4, 14.0, now)

    active_tracks = [trk1, trk2, trk3]

    # Mock detection container for boxes
    class MockBox:
        def __init__(self, xyxy, cls_id, conf):
            self.xyxy = [np.array(xyxy, dtype=float)]
            self.cls = cls_id
            self.conf = conf

    class MockDets:
        def __init__(self):
            self.boxes = [
                MockBox((270, 240, 370, 330), 2, 0.88),
                MockBox((460, 260, 500, 340), 0, 0.82),
                MockBox((120, 250, 210, 320), 19, 0.91),
            ]
            self.names = {0: 'person', 2: 'car', 19: 'cow'}

    mock_dets = MockDets()

    # Pothole
    potholes = [{
        'bbox': (300, 360, 55, 32),
        'center': (327, 376),
        'area': 5200,
        'depth': 34.0,
        'confidence': 0.85,
        'severity': 'MODERATE',
        'ground_pos': (0.1, 4.2)
    }]

    # Costmap planning
    plan_result = costmap_planner.plan(road_result, active_tracks, potholes, current_speed=1.5)

    # State & control
    state = "YIELD CROSSING"
    steering_angle = stanley_controller.compute_steering(
        plan_result['target_lateral_offset_m'],
        plan_result['target_angle_deg'],
        current_speed_ms=1.5
    )
    throttle = 25

    warnings = [
        "CROSSING ALERT: Person approaching from right (11.2m)",
        "POTHOLE -8.5cm DETECTED AHEAD (4.2m)"
    ]

    # Render HUD
    vis = frame.copy()
    draw_3d_trajectory_ribbon(vis, road_result, plan_result, camera_geometry, steering_angle)
    draw_nextgen_detections(vis, mock_dets, active_tracks, camera_geometry, warnings)
    draw_topographical_potholes(vis, potholes)
    draw_state_and_alerts(vis, state, warnings)
    draw_cockpit_dashboard(vis, 31.0, len(mock_dets.boxes), len(potholes), state, steering_angle, throttle)
    draw_tactical_bev_radar(vis, plan_result)

    out_file = 'test_hud_output.jpg'
    cv2.imwrite(out_file, vis)
    print(f"Successfully generated next-gen demo HUD frame to {out_file} ({w}x{h})")

if __name__ == '__main__':
    main()
