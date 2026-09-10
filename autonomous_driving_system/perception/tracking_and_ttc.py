"""
Multi-Object Tracker, Velocity Estimator, and Instantaneous Time-To-Collision (iTTC) Node
Calibrated for Indian Traffic Conditions (pedestrians darting across, stray cattle, unlaned vehicles).

Fuses:
  - F1TENTH / Autonomous-Racing Safety Node (iTTC calculation)
  - Autoware Tier4 Dynamic Obstacle Crossing Interception
  - OpenADAS metric trajectory tracking
"""

import time
import numpy as np
from collections import deque


class TrackedObject:
    def __init__(self, track_id, bbox, cls_id, name, X, Z, timestamp):
        self.track_id = track_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.cls_id = cls_id
        self.name = name
        self.X = X  # Lateral position in meters
        self.Z = Z  # Forward distance in meters
        self.last_seen = timestamp
        self.first_seen = timestamp
        self.hits = 1
        
        # History for velocity smoothing
        self.history = deque(maxlen=6)
        self.history.append((X, Z, timestamp))
        
        # Estimated velocities (m/s)
        self.vz = 0.0  # Closing rate: positive means closing in towards ego-vehicle
        self.vx = 0.0  # Lateral rate: positive means moving right, negative means moving left
        self.ittc = 99.0  # Instantaneous Time-To-Collision in seconds
        
        # Hazard status
        self.is_closing = False
        self.is_crossing = False
        self.crossing_dir = None  # 'LEFT_TO_RIGHT' or 'RIGHT_TO_LEFT'
        self.crossing_time = 99.0

    def update(self, bbox, X, Z, timestamp):
        dt = timestamp - self.last_seen
        if dt > 0.001:
            # Raw instant velocity
            raw_vz = (self.Z - Z) / dt  # Positive = closing in
            raw_vx = (X - self.X) / dt  # Positive = moving right

            # Low-pass filter velocity with history
            alpha = 0.45
            self.vz = alpha * raw_vz + (1 - alpha) * self.vz
            self.vx = alpha * raw_vx + (1 - alpha) * self.vx

            # Instantaneous Time-To-Collision (iTTC) calculation
            if self.vz > 0.35:  # Closing in with meaningful speed
                self.is_closing = True
                self.ittc = max(0.05, Z / max(self.vz, 0.05))
            else:
                self.is_closing = False
                self.ittc = 99.0

            # Crossing Interception Detection:
            # If object is laterally outside vehicle body (|X| > 0.7m) but moving toward center
            corridor_half_width = 1.0  # meters
            if abs(X) > 0.6:
                is_heading_inward = (X > 0.6 and self.vx < -0.3) or (X < -0.6 and self.vx > 0.3)
                if is_heading_inward:
                    # Time to reach ego-lane corridor edge
                    dist_to_corridor = max(0.0, abs(X) - corridor_half_width)
                    time_to_corridor = dist_to_corridor / max(abs(self.vx), 0.1)
                    if time_to_corridor < 2.5 and Z < 25.0:
                        self.is_crossing = True
                        self.crossing_dir = 'RIGHT_TO_LEFT' if X > 0 else 'LEFT_TO_RIGHT'
                        self.crossing_time = time_to_corridor
                    else:
                        self.is_crossing = False
                else:
                    self.is_crossing = False
            else:
                self.is_crossing = False

        self.bbox = bbox
        self.X = X
        self.Z = Z
        self.last_seen = timestamp
        self.hits += 1
        self.history.append((X, Z, timestamp))


class TrackingAndTTC:
    """
    Maintains persistent obstacle tracks, calculates closing velocity & iTTC,
    and flags anticipatory collision/crossing hazards.
    """

    def __init__(self, camera_geometry, track_thresh_dist=2.2, max_stale_time=0.45):
        self.camera_geometry = camera_geometry
        self.track_thresh_dist = track_thresh_dist  # Distance threshold (meters) for spatial matching
        self.max_stale_time = max_stale_time        # Seconds before dropping track
        
        self.tracks = {}
        self.next_track_id = 1
        
        # iTTC Safety Thresholds (seconds)
        self.TTC_CRITICAL_VEHICLE = 1.65  # Emergency stop threshold
        self.TTC_WARNING_VEHICLE = 2.80   # Slow down threshold
        self.TTC_CRITICAL_VULNERABLE = 2.00  # Higher margin for pedestrians, cows, dogs
        self.TTC_WARNING_VULNERABLE = 3.50

        # Vulnerable Indian road classes (pedestrians, bikes, cows, dogs, goats)
        self.vulnerable_classes = {0, 1, 3, 15, 16, 17, 18, 19, 20}

    def update(self, detections, current_time=None):
        """
        Updates object tracker with current frame detections.
        
        Args:
            detections: YOLO Results object
            current_time: timestamp (float)
        
        Returns:
            tracked_objects: List of TrackedObject instances
            safety_alerts: List of structured hazard dicts:
                {
                   'type': 'AEB' | 'TTC_WARN' | 'CROSSING_ALERT' | 'PROXIMITY',
                   'track_id': int,
                   'name': str,
                   'distance': float,
                   'ittc': float,
                   'details': str
                }
        """
        if current_time is None:
            current_time = time.time()

        det_items = []
        if detections and detections.boxes is not None:
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls)
                conf = float(box.conf)
                if conf < 0.35:
                    continue

                from perception.object_detector import ObjectDetector
                name = ObjectDetector.get_india_name(cls_id, detections.names.get(cls_id, ''))
                
                # Estimate 3D position (X, Z) in meters
                X, Z = self.camera_geometry.estimate_object_position((x1, y1, x2, y2), cls_id)
                det_items.append({
                    'bbox': (x1, y1, x2, y2),
                    'cls_id': cls_id,
                    'name': name,
                    'conf': conf,
                    'X': X,
                    'Z': Z
                })

        # Match detections to existing tracks by Euclidean distance in 3D ground space
        matched_tracks = set()
        unmatched_dets = []

        for det in det_items:
            best_id = None
            best_score = float('inf')
            
            for tid, trk in self.tracks.items():
                if tid in matched_tracks:
                    continue
                dt = max(0.01, current_time - trk.last_seen)
                # Dynamic matching threshold scales with maximum possible vehicle velocity (up to 30 m/s)
                max_allowed_dist = max(self.track_thresh_dist, 30.0 * dt)
                
                # Spatial distance between detection and predicted or previous track position
                pred_Z = trk.Z - trk.vz * dt if trk.hits > 1 else trk.Z
                pred_X = trk.X + trk.vx * dt if trk.hits > 1 else trk.X
                dist = np.hypot(det['X'] - pred_X, det['Z'] - pred_Z)

                # Also check raw distance
                raw_dist = np.hypot(det['X'] - trk.X, det['Z'] - trk.Z)
                min_dist = min(dist, raw_dist)

                if min_dist < max_allowed_dist and (det['cls_id'] == trk.cls_id or trk.hits < 2):
                    if min_dist < best_score:
                        best_score = min_dist
                        best_id = tid

            if best_id is not None:
                self.tracks[best_id].update(det['bbox'], det['X'], det['Z'], current_time)
                matched_tracks.add(best_id)
            else:
                unmatched_dets.append(det)

        # Create new tracks for unmatched detections
        for det in unmatched_dets:
            new_trk = TrackedObject(
                self.next_track_id,
                det['bbox'],
                det['cls_id'],
                det['name'],
                det['X'],
                det['Z'],
                current_time
            )
            self.tracks[self.next_track_id] = new_trk
            self.next_track_id += 1

        # Remove stale tracks
        stale_ids = [
            tid for tid, trk in self.tracks.items()
            if (current_time - trk.last_seen) > self.max_stale_time
        ]
        for tid in stale_ids:
            del self.tracks[tid]

        # Evaluate safety alerts & iTTC hazards
        safety_alerts = []
        active_tracks = list(self.tracks.values())

        for trk in active_tracks:
            # Ignore targets not seen in current frame
            if (current_time - trk.last_seen) > 0.15:
                continue

            is_vulnerable = trk.cls_id in self.vulnerable_classes
            ttc_crit = self.TTC_CRITICAL_VULNERABLE if is_vulnerable else self.TTC_CRITICAL_VEHICLE
            ttc_warn = self.TTC_WARNING_VULNERABLE if is_vulnerable else self.TTC_WARNING_VEHICLE

            # 1. Critical iTTC -> Automatic Emergency Braking (AEB)
            # Applies if directly in front corridor (|X| < 1.35m) and closing in fast
            if abs(trk.X) < 1.35 and trk.is_closing and trk.ittc < ttc_crit:
                safety_alerts.append({
                    'type': 'AEB',
                    'track_id': trk.track_id,
                    'name': trk.name,
                    'distance': trk.Z,
                    'ittc': trk.ittc,
                    'details': f"CRITICAL AEB: {trk.name} ({trk.Z:.1f}m, TTC {trk.ittc:.1f}s)"
                })
            # 2. Warning iTTC -> Decelerate / Prepare evasive
            elif abs(trk.X) < 1.45 and trk.is_closing and trk.ittc < ttc_warn:
                safety_alerts.append({
                    'type': 'TTC_WARN',
                    'track_id': trk.track_id,
                    'name': trk.name,
                    'distance': trk.Z,
                    'ittc': trk.ittc,
                    'details': f"WARNING: {trk.name} closing ({trk.Z:.1f}m, TTC {trk.ittc:.1f}s)"
                })
            # 3. Sudden Crossing Interception (Indian Roads Pedestrians / Cattle darting)
            elif trk.is_crossing:
                safety_alerts.append({
                    'type': 'CROSSING_ALERT',
                    'track_id': trk.track_id,
                    'name': trk.name,
                    'distance': trk.Z,
                    'ittc': trk.crossing_time,
                    'details': f"CROSSING ALERT: {trk.name} approaching from {trk.crossing_dir} ({trk.Z:.1f}m)"
                })
            # 4. Proximity alert (near vehicle regardless of speed)
            elif abs(trk.X) < 1.25 and trk.Z < (4.0 if is_vulnerable else 5.5):
                safety_alerts.append({
                    'type': 'PROXIMITY',
                    'track_id': trk.track_id,
                    'name': trk.name,
                    'distance': trk.Z,
                    'ittc': trk.ittc,
                    'details': f"PROXIMITY: {trk.name} close ({trk.Z:.1f}m)"
                })

        return active_tracks, safety_alerts
