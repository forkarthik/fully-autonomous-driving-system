"""
Costmap Fusion and Dynamic Corridor Planner (Follow-The-Gap & Pothole Dodge)
Inspired by:
  - F1TENTH / Autonomous-Racing Reactive Gap Follow (Disparity Extender)
  - Autoware Tier4 Dynamic Obstacle Avoidance & Costmap Fusion
  - Swaayatt Robots unlaned road corridor negotiation

Fuses:
  1. Drivable road surface boundaries (from LaneDetector)
  2. Dynamic tracked obstacles with safety bubbles (from TrackingAndTTC)
  3. Confirmed road potholes with severity weights (from PotholeDetector)
"""

import numpy as np
import cv2


class CostmapPlanner:
    """
    Computes optimal free-space clearance corridor and collision-free waypoints.
    """

    def __init__(self, frame_width=640, frame_height=480, camera_geometry=None):
        self.w = frame_width
        self.h = frame_height
        self.camera_geometry = camera_geometry

        # Angular scan parameters (-45 deg to +45 deg ahead of vehicle)
        self.num_rays = 61
        self.min_angle_deg = -45.0
        self.max_angle_deg = 45.0
        self.angles_deg = np.linspace(self.min_angle_deg, self.max_angle_deg, self.num_rays)
        self.angles_rad = np.radians(self.angles_deg)

        # Maximum lookahead planning horizon (meters)
        self.max_range = 25.0
        self.min_safe_clearance = 1.6  # Minimum gap width in meters for ego-vehicle

        # Class safety bubble radii (meters)
        # Unpredictable Indian road elements get larger safety margins
        self.safety_bubbles = {
            0: 1.2,   # Pedestrian (can suddenly change direction)
            1: 1.0,   # Bicycle / cycle-rickshaw
            2: 1.3,   # Auto-rickshaw / Car
            3: 0.9,   # Motorcycle / Scooter
            5: 2.0,   # Bus
            7: 2.0,   # Truck / Lorry
            15: 0.8,  # Cat
            16: 1.2,  # Stray Dog
            18: 1.0,  # Goat
            19: 1.8,  # Cow / Bull (large, unpredictable)
            20: 2.5,  # Elephant
            'default': 1.0,
            'pothole': 0.85
        }

        # Ego vehicle physical footprint
        self.car_width = 1.0   # meters
        self.car_length = 1.4  # meters

    def plan(self, road_result, tracked_objects, potholes, current_speed=1.5):
        """
        Executes Follow-the-Gap and Pothole Avoidance trajectory planning.

        Args:
            road_result: dict from LaneDetector with 'road_mask' and 'lanes_info'
            tracked_objects: list of TrackedObject from TrackingAndTTC
            potholes: list of confirmed pothole dicts from PotholeDetector
            current_speed: estimated vehicle speed (m/s)

        Returns:
            plan_result: dict containing:
                - 'target_x_px': pixel x coordinate of target steering aim
                - 'target_lateral_offset_m': lateral target offset in meters (0=center, +right, -left)
                - 'target_angle_deg': recommended heading angle
                - 'best_gap': (start_angle_deg, end_angle_deg)
                - 'suggested_state': 'CRUISING' | 'GAP_AVOID_LEFT' | 'GAP_AVOID_RIGHT' | 'DODGE_POTHOLE' | 'SLOW'
                - 'scan_ranges': 1D array of free-space distances across angles
                - 'bev_occupancy': local 2D bird's-eye occupancy grid image for visualization
        """
        # 1. Initialize ray distances to maximum horizon
        ranges = np.full(self.num_rays, self.max_range, dtype=np.float32)

        # 2. Bound rays by the drivable road boundaries
        lanes_info = road_result.get('lanes_info', {}) if road_result else {}
        lx = lanes_info.get('left_x')
        rx = lanes_info.get('right_x')
        center_x = self.w // 2

        # Convert road pixel boundaries to approximate angular limits
        if lx is not None:
            left_offset_px = lx - center_x
            # Negative angle limit for left side
            left_angle_limit = np.degrees(np.arctan2(left_offset_px * 0.005, 1.0))
            for i, ang in enumerate(self.angles_deg):
                if ang < left_angle_limit:
                    ranges[i] = min(ranges[i], 1.5)  # Constrained by left road edge

        if rx is not None:
            right_offset_px = rx - center_x
            right_angle_limit = np.degrees(np.arctan2(right_offset_px * 0.005, 1.0))
            for i, ang in enumerate(self.angles_deg):
                if ang > right_angle_limit:
                    ranges[i] = min(ranges[i], 1.5)  # Constrained by right road edge

        # 3. Apply Obstacles & Safety Bubbles (Disparity Extender)
        nearest_obstacle_dist = float('inf')
        nearest_obstacle_x = 0.0

        for trk in tracked_objects:
            # Only consider forward objects
            if trk.Z <= 0.5 or trk.Z > self.max_range:
                continue

            dist = np.hypot(trk.X, trk.Z)
            if dist < nearest_obstacle_dist:
                nearest_obstacle_dist = dist
                nearest_obstacle_x = trk.X

            bubble_r = self.safety_bubbles.get(trk.cls_id, self.safety_bubbles['default'])
            # Expand bubble if object is actively crossing
            if trk.is_crossing:
                bubble_r *= 1.35

            # Angular width of obstacle bubble
            obj_angle_rad = np.arctan2(trk.X, trk.Z)
            angular_span_rad = np.arcsin(min(0.95, bubble_r / max(dist, bubble_r + 0.01)))

            min_blocked_rad = obj_angle_rad - angular_span_rad
            max_blocked_rad = obj_angle_rad + angular_span_rad

            # Mask out ranges inside bubble
            for i, ang_rad in enumerate(self.angles_rad):
                if min_blocked_rad <= ang_rad <= max_blocked_rad:
                    # Point blocked at distance (dist - bubble_r)
                    safe_d = max(0.0, dist - bubble_r)
                    ranges[i] = min(ranges[i], safe_d)

        # 4. Apply Potholes to Cost Ranges
        pothole_threat = None
        for p in potholes:
            severity = p.get('severity', 'MINOR')
            if severity in ('MODERATE', 'SEVERE', 'CRITICAL'):
                gx, gz = p.get('ground_pos', (0.0, 5.0))
                if 0.5 < gz < 18.0:
                    dist = np.hypot(gx, gz)
                    bubble_r = 1.1 if severity == 'CRITICAL' else 0.85 if severity == 'SEVERE' else 0.65

                    # If pothole is directly in front path, mark for avoidance
                    if abs(gx) < 0.85 and (pothole_threat is None or gz < pothole_threat['gz']):
                        pothole_threat = {'gx': gx, 'gz': gz, 'severity': severity, 'bubble': bubble_r}

                    p_angle_rad = np.arctan2(gx, gz)
                    p_span_rad = np.arcsin(min(0.95, bubble_r / max(dist, bubble_r + 0.01)))
                    for i, ang_rad in enumerate(self.angles_rad):
                        if (p_angle_rad - p_span_rad) <= ang_rad <= (p_angle_rad + p_span_rad):
                            ranges[i] = min(ranges[i], max(0.0, dist - bubble_r))

        # 5. Find Max Free Space Gap
        # A gap is a contiguous sequence of angles where range >= threshold
        gap_threshold = 4.0  # meters of forward clearance required
        gaps = []
        current_gap_start = None

        for i, r in enumerate(ranges):
            if r >= gap_threshold:
                if current_gap_start is None:
                    current_gap_start = i
            else:
                if current_gap_start is not None:
                    gaps.append((current_gap_start, i - 1))
                    current_gap_start = None

        if current_gap_start is not None:
            gaps.append((current_gap_start, len(ranges) - 1))

        # 6. Select Best Gap and Aiming Point
        best_point_idx = self.num_rays // 2  # Default: straight ahead
        suggested_state = "CRUISING"

        if gaps:
            # Score each gap: width * depth, plus bias towards center
            def score_gap(gap):
                start_i, end_i = gap
                gap_width = end_i - start_i + 1
                avg_depth = np.mean(ranges[start_i:end_i + 1])
                mid_angle = (self.angles_deg[start_i] + self.angles_deg[end_i]) / 2.0
                center_penalty = abs(mid_angle) * 0.15
                return (gap_width * 1.5) + (avg_depth * 1.0) - center_penalty

            best_gap = max(gaps, key=score_gap)
            start_i, end_i = best_gap
            
            # Select center or deepest point in the gap
            gap_ranges = ranges[start_i:end_i + 1]
            deepest_local_idx = np.argmax(gap_ranges)
            # Blend midpoint of gap with deepest point for smoother steering
            mid_idx = (start_i + end_i) // 2
            best_point_idx = int(round(0.6 * mid_idx + 0.4 * (start_i + deepest_local_idx)))
            best_gap_deg = (self.angles_deg[start_i], self.angles_deg[end_i])
        else:
            # If all gaps are constrained, find max range angle
            best_point_idx = int(np.argmax(ranges))
            best_gap_deg = (self.angles_deg[best_point_idx], self.angles_deg[best_point_idx])
            suggested_state = "SLOW"

        target_angle_deg = float(self.angles_deg[best_point_idx])
        target_angle_rad = np.radians(target_angle_deg)
        lookahead_dist = min(8.0, max(2.5, float(ranges[best_point_idx])))

        # Compute lateral target offset in meters
        target_lateral_offset_m = float(lookahead_dist * np.sin(target_angle_rad))

        # Determine suggested behavioral state
        if pothole_threat is not None and pothole_threat['gz'] < 10.0:
            suggested_state = "DODGE POTHOLE"
        elif target_angle_deg < -8.0:
            suggested_state = "AVOID LEFT"
        elif target_angle_deg > 8.0:
            suggested_state = "AVOID RIGHT"
        elif suggested_state != "SLOW":
            suggested_state = "CRUISING"

        # Convert target angle to pixel coordinate on 640x480 frame
        # Focal projection: u = cx + fx * tan(angle)
        fx = self.w * 0.95
        target_x_px = int(self.w / 2.0 + fx * np.tan(target_angle_rad))
        target_x_px = max(20, min(self.w - 20, target_x_px))

        # 7. Render Bird's-Eye View (BEV) Mini-Radar
        bev_img = self._render_bev_radar(ranges, best_point_idx, tracked_objects, potholes)

        return {
            'target_x_px': target_x_px,
            'target_lateral_offset_m': target_lateral_offset_m,
            'target_angle_deg': target_angle_deg,
            'best_gap': best_gap_deg,
            'suggested_state': suggested_state,
            'scan_ranges': ranges,
            'bev_occupancy': bev_img
        }

    def _render_bev_radar(self, ranges, target_idx, tracked_objects, potholes):
        """
        Renders a sleek, high-tech Bird's-Eye View (BEV) mini-radar.
        Can be overlaid into the top corner of the HUD.
        """
        radar_size = 140
        radar = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
        radar[:] = (18, 18, 22)  # Sleek dark carbon background

        center_x = radar_size // 2
        bottom_y = radar_size - 12
        max_radar_m = 20.0
        scale = (radar_size - 25) / max_radar_m  # Pixels per meter

        # Concentric range rings (5m, 10m, 15m)
        for r_m in [5, 10, 15]:
            r_px = int(r_m * scale)
            cv2.ellipse(radar, (center_x, bottom_y), (r_px, r_px), 0, 180, 360, (40, 45, 55), 1, cv2.LINE_AA)

        # Free space corridor fan
        pts = [(center_x, bottom_y)]
        for i, ang_deg in enumerate(self.angles_deg):
            r = min(max_radar_m, ranges[i])
            rad = np.radians(ang_deg)
            # Vehicle frame: +X right, +Z forward
            # Image radar: +X right, -Y upward
            px = int(center_x + r * scale * np.sin(rad))
            py = int(bottom_y - r * scale * np.cos(rad))
            pts.append((px, py))

        if len(pts) > 2:
            poly_pts = np.array([pts], dtype=np.int32)
            overlay = radar.copy()
            cv2.fillPoly(overlay, poly_pts, (40, 110, 50))  # Translucent green free-space
            cv2.addWeighted(overlay, 0.45, radar, 0.55, 0, radar)

        # Draw Potholes on Radar (Orange/Red circles)
        for p in potholes:
            gx, gz = p.get('ground_pos', (0.0, 0.0))
            if 0.5 < gz < max_radar_m:
                px = int(center_x + gx * scale)
                py = int(bottom_y - gz * scale)
                if 2 <= px < radar_size - 2 and 2 <= py < radar_size - 2:
                    pcolor = (50, 60, 240) if p.get('severity') in ('CRITICAL', 'SEVERE') else (50, 160, 255)
                    cv2.circle(radar, (px, py), 3, pcolor, -1, cv2.LINE_AA)

        # Draw Tracked Obstacles on Radar
        for trk in tracked_objects:
            if 0.5 < trk.Z < max_radar_m:
                px = int(center_x + trk.X * scale)
                py = int(bottom_y - trk.Z * scale)
                if 2 <= px < radar_size - 2 and 2 <= py < radar_size - 2:
                    color = (200, 60, 220) if trk.cls_id == 0 else (230, 220, 50) if trk.cls_id in (15, 16, 17, 18, 19) else (72, 225, 100)
                    cv2.circle(radar, (px, py), 4, color, -1, cv2.LINE_AA)

                    # Draw velocity vector line if moving
                    if abs(trk.vx) > 0.4 or abs(trk.vz) > 0.4:
                        vpx = int(px + trk.vx * scale * 0.8)
                        vpy = int(py + trk.vz * scale * 0.8)
                        cv2.line(radar, (px, py), (vpx, vpy), (255, 255, 255), 1, cv2.LINE_AA)

        # Target Trajectory Steering Ray (Yellow line)
        target_deg = self.angles_deg[target_idx]
        target_rad = np.radians(target_deg)
        t_range = min(max_radar_m, ranges[target_idx])
        t_px = int(center_x + t_range * scale * np.sin(target_rad))
        t_py = int(bottom_y - t_range * scale * np.cos(target_rad))
        cv2.line(radar, (center_x, bottom_y), (t_px, t_py), (50, 220, 245), 2, cv2.LINE_AA)
        cv2.circle(radar, (t_px, t_py), 3, (50, 220, 245), -1, cv2.LINE_AA)

        # Ego Vehicle icon (Cyan rectangle at bottom)
        ego_w = int(self.car_width * scale)
        ego_l = int(self.car_length * scale)
        cv2.rectangle(radar,
                      (center_x - ego_w // 2, bottom_y - ego_l),
                      (center_x + ego_w // 2, bottom_y),
                      (230, 220, 50), -1)

        # Outer border
        cv2.rectangle(radar, (0, 0), (radar_size - 1, radar_size - 1), (60, 65, 80), 1)
        cv2.putText(radar, "BEV RADAR", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        return radar
