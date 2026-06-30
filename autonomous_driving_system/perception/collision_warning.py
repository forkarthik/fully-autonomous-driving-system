import cv2
import numpy as np


class CollisionWarning:
    """
    Collision warning system calibrated for Indian road conditions.
    
    Handles Indian vehicles (auto-rickshaws, bikes, lorries, pushcarts)
    and common road animals (cows, dogs, goats).
    """

    def __init__(self, frame_shape):
        self.h, self.w = frame_shape[:2]

        # ROI: Trapezoid representing the ego-lane ahead
        self.roi_points = np.array([
            [0.15, 1.0],   # Bottom Left  (wider for Indian roads)
            [0.85, 1.0],   # Bottom Right
            [0.60, 0.55],  # Top Right
            [0.40, 0.55],  # Top Left
        ], dtype=np.float32)
        self.roi_pixel = (self.roi_points * [self.w, self.h]).astype(np.int32)

        # Distance estimation — focal length approximation
        self.focal_length = self.w * 0.9

        # Real-world widths (meters) for Indian vehicles
        self.real_widths = {
            0:  0.5,   # Person
            1:  0.6,   # Bicycle / Cycle rickshaw
            2:  1.6,   # Car / Auto-rickshaw (smaller than western cars)
            3:  0.7,   # Motorcycle / Scooter
            5:  2.5,   # Bus
            7:  2.5,   # Lorry / Truck
            15: 0.3,   # Cat
            16: 0.4,   # Dog
            17: 1.5,   # Horse
            18: 0.5,   # Sheep / Goat
            19: 1.8,   # Cow / Bull
            20: 3.0,   # Elephant
            'default': 0.8,
        }

        # Road sign class IDs
        self.stop_sign_id = 11
        self.traffic_light_id = 9

        # All obstacle classes (vehicles + persons + animals)
        self.obstacle_classes = [0, 1, 2, 3, 5, 7, 15, 16, 17, 18, 19, 20]

        # Distance thresholds
        self.critical_dist = 8.0    # meters
        self.warning_dist = 20.0    # meters
        self.animal_critical_dist = 5.0   # Animals are unpredictable — closer threshold
        self.animal_warning_dist = 12.0

    def estimate_distance(self, box_w, cls_id):
        """Estimate distance using pinhole camera model: D = (W_real * f) / W_pixel"""
        if box_w <= 0:
            return 999.0
        w_real = self.real_widths.get(cls_id, self.real_widths['default'])
        return (w_real * self.focal_length) / box_w

    def detect_light_color(self, roi):
        """Detect traffic light color from its ROI."""
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        area = roi.shape[0] * roi.shape[1]

        # HSV ranges
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

        # Safety priority: if red is significant, call it red
        if r_count > threshold and r_count >= g_count * 0.7:
            return "RED LIGHT"

        return best

    def _is_animal(self, cls_id):
        return cls_id in [15, 16, 17, 18, 19, 20]

    def check_collision(self, detections, frame):
        """
        Check for collisions, traffic signs, and road hazards.
        
        Returns:
            warnings: list of warning strings
            roi_color: BGR tuple for ROI visualization
        """
        warnings = []
        roi_color = (0, 255, 0)

        if not detections:
            return warnings, roi_color

        max_hazard = 0  # 0=Safe, 1=Warning, 2=Critical

        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            conf = float(box.conf)
            box_w = x2 - x1
            box_h = y2 - y1

            # --- Traffic signs ---
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

            # --- Obstacles (vehicles, persons, animals) ---
            if cls in self.obstacle_classes:
                cx = (x1 + x2) // 2
                cy = y2  # Bottom center

                is_inside = cv2.pointPolygonTest(self.roi_pixel, (cx, cy), False) >= 0

                if is_inside:
                    dist = self.estimate_distance(box_w, cls)

                    if self._is_animal(cls):
                        crit_d = self.animal_critical_dist
                        warn_d = self.animal_warning_dist
                    else:
                        crit_d = self.critical_dist
                        warn_d = self.warning_dist

                    # Get display name
                    from perception.object_detector import ObjectDetector
                    name = ObjectDetector.get_india_name(cls, detections.names.get(cls, ''))

                    if dist < crit_d:
                        warnings.append(f"CRITICAL: {name} ({dist:.1f}m)")
                        max_hazard = max(max_hazard, 2)
                    elif dist < warn_d:
                        warnings.append(f"WARNING: {name} ({dist:.1f}m)")
                        max_hazard = max(max_hazard, 1)

        if max_hazard == 2:
            roi_color = (0, 0, 255)
        elif max_hazard == 1:
            roi_color = (0, 255, 255)

        return list(set(warnings)), roi_color

    def draw_roi(self, frame, color=(0, 255, 0), thickness=2):
        """Draw the ROI polygon with translucent fill."""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.roi_pixel], color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [self.roi_pixel], True, color, thickness, cv2.LINE_AA)
        return frame
