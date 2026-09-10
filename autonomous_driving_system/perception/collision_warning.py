import cv2
import numpy as np


class CollisionWarning:
    """
    Advanced Collision Warning & Threat Assessment System.
    Calibrated for Indian road conditions (unlaned traffic, auto-rickshaws, bikes,
    cows, dogs, and crossing pedestrians).

    Integrates:
      - Real-world 3D ground projection & pinhole estimation
      - Instantaneous Time-To-Collision (iTTC) & Automatic Emergency Braking (AEB)
      - Anticipatory crossing interception alerts
      - Traffic sign and color-aware traffic light detection
    """

    def __init__(self, frame_shape, camera_geometry=None):
        self.h, self.w = frame_shape[:2]
        self.camera_geometry = camera_geometry

        # ROI: Dynamic trapezoid representing the ego-corridor ahead
        self.roi_points = np.array([
            [0.15, 1.00],   # Bottom Left
            [0.85, 1.00],   # Bottom Right
            [0.60, 0.52],   # Top Right
            [0.40, 0.52],   # Top Left
        ], dtype=np.float32)
        self.roi_pixel = (self.roi_points * [self.w, self.h]).astype(np.int32)

        # Distance estimation fallback focal length
        self.focal_length = self.w * 0.95

        # Real-world widths (meters) for Indian vehicles
        self.real_widths = {
            0:  0.50,   # Person
            1:  0.60,   # Bicycle / Cycle rickshaw
            2:  1.65,   # Car / Auto-rickshaw
            3:  0.70,   # Motorcycle / Scooter
            5:  2.50,   # Bus
            7:  2.50,   # Lorry / Truck
            15: 0.35,   # Cat
            16: 0.45,   # Dog
            17: 1.50,   # Horse
            18: 0.55,   # Sheep / Goat
            19: 1.80,   # Cow / Bull
            20: 3.00,   # Elephant
            'default': 1.0,
        }

        # Road sign class IDs
        self.stop_sign_id = 11
        self.traffic_light_id = 9

        # Obstacle classes (vehicles + persons + animals)
        self.obstacle_classes = [0, 1, 2, 3, 5, 7, 15, 16, 17, 18, 19, 20]

        # Distance thresholds (meters)
        self.critical_dist = 7.5
        self.warning_dist = 18.0
        self.animal_critical_dist = 6.0   # Unpredictable animals have stricter margin
        self.animal_warning_dist = 14.0

    def estimate_distance(self, box_w, cls_id, bbox=None):
        """Estimate distance using CameraGeometry ground projection or pinhole fallback."""
        if bbox is not None and self.camera_geometry is not None:
            _, Z = self.camera_geometry.estimate_object_position(bbox, cls_id)
            return Z
        if box_w <= 0:
            return 999.0
        w_real = self.real_widths.get(cls_id, self.real_widths['default'])
        return float((w_real * self.focal_length) / box_w)

    def detect_light_color(self, roi):
        """Detect traffic light color from its ROI."""
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        area = roi.shape[0] * roi.shape[1]

        mask_r1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([12, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([160, 60, 60]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)
        mask_yellow = cv2.inRange(hsv, np.array([15, 60, 60]), np.array([35, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([36, 50, 50]), np.array([90, 255, 255]))

        r_count = cv2.countNonZero(mask_red)
        y_count = cv2.countNonZero(mask_yellow)
        g_count = cv2.countNonZero(mask_green)

        threshold = area * 0.03
        counts = {'RED LIGHT': r_count, 'YELLOW LIGHT': y_count, 'GREEN LIGHT': g_count}
        best = max(counts, key=counts.get)

        if counts[best] < threshold:
            return None

        # Safety priority: red is prioritized
        if r_count > threshold and r_count >= g_count * 0.7:
            return "RED LIGHT"

        return best

    def _is_animal(self, cls_id):
        return cls_id in [15, 16, 17, 18, 19, 20]

    def check_collision(self, detections, frame, tracker_alerts=None):
        """
        Evaluate road safety combining static distance ROI, dynamic iTTC, and traffic signals.
        
        Args:
            detections: YOLO detection results
            frame: input BGR frame
            tracker_alerts: optional alerts list from TrackingAndTTC
        
        Returns:
            warnings: list of prioritized warning strings
            roi_color: BGR tuple for ROI visualization
        """
        warnings = []
        roi_color = (0, 255, 0)
        max_hazard = 0  # 0=Safe, 1=Warning, 2=Critical

        # 1. Incorporate Dynamic Tracking & iTTC Alerts
        if tracker_alerts:
            for alert in tracker_alerts:
                atype = alert.get('type')
                details = alert.get('details', '')
                if atype == 'AEB':
                    warnings.append(details)
                    max_hazard = max(max_hazard, 2)
                elif atype == 'TTC_WARN':
                    warnings.append(details)
                    max_hazard = max(max_hazard, 1)
                elif atype == 'CROSSING_ALERT':
                    warnings.append(details)
                    max_hazard = max(max_hazard, 1)
                elif atype == 'PROXIMITY':
                    warnings.append(details)
                    max_hazard = max(max_hazard, 2 if alert.get('distance', 99) < 4.0 else 1)

        # 2. Check Static Detections (Traffic lights, stop signs, static obstacles)
        if detections and detections.boxes is not None:
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = float(box.conf)
                box_w = x2 - x1
                box_h = y2 - y1

                # Traffic signs
                if cls == self.stop_sign_id and conf > 0.45:
                    if box_h > self.h * 0.04:
                        warnings.append("STOP SIGN")

                elif cls == self.traffic_light_id and conf > 0.40:
                    if box_h > self.h * 0.03:
                        tl_roi = frame[max(0, y1):min(self.h, y2), max(0, x1):min(self.w, x2)]
                        if tl_roi.size > 0:
                            color = self.detect_light_color(tl_roi)
                            if color:
                                warnings.append(color)

                # Obstacle proximity check
                if cls in self.obstacle_classes:
                    cx = (x1 + x2) // 2
                    cy = y2
                    is_inside = cv2.pointPolygonTest(self.roi_pixel, (cx, cy), False) >= 0

                    if is_inside:
                        dist = self.estimate_distance(box_w, cls, bbox=(x1, y1, x2, y2))
                        crit_d = self.animal_critical_dist if self._is_animal(cls) else self.critical_dist
                        warn_d = self.animal_warning_dist if self._is_animal(cls) else self.warning_dist

                        from perception.object_detector import ObjectDetector
                        name = ObjectDetector.get_india_name(cls, detections.names.get(cls, ''))

                        if dist < crit_d:
                            warnings.append(f"CRITICAL: {name} ({dist:.1f}m)")
                            max_hazard = max(max_hazard, 2)
                        elif dist < warn_d:
                            warnings.append(f"WARNING: {name} ({dist:.1f}m)")
                            max_hazard = max(max_hazard, 1)

        if max_hazard == 2:
            roi_color = (60, 60, 240)   # Red
        elif max_hazard == 1:
            roi_color = (50, 220, 245)  # Yellow/Orange

        # Deduplicate preserving order
        unique_warnings = []
        for w in warnings:
            if w not in unique_warnings:
                unique_warnings.append(w)

        return unique_warnings, roi_color

    def draw_roi(self, frame, color=(0, 255, 0), thickness=2):
        """Draw the ROI polygon with translucent fill."""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.roi_pixel], color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [self.roi_pixel], True, color, thickness, cv2.LINE_AA)
        return frame
