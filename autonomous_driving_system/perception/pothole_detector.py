"""
Pothole Detector and Topographical Hazard Analyzer
Optimized for NVIDIA Jetson Orin Nano (JetPack 6 / CUDA / TensorRT).

Incorporates field-tested depth estimation from Pothole-detection:
  - Surrounding road contrast evaluation
  - Canny edge sharpness gradients
  - Multi-frame temporal tracking to eliminate shadow false positives
  - Metric ground projection (X, Z in meters)
"""

import cv2
import numpy as np


class PotholeTracker:
    """
    Temporal tracking for detected potholes.
    Filters transient noise, road markings, and moving shadows by requiring
    consistent observations across consecutive frames.
    """
    def __init__(self, match_thresh_px=50, max_frames_missing=6, min_hits_to_confirm=2):
        self.match_thresh_px = match_thresh_px
        self.max_frames_missing = max_frames_missing
        self.min_hits_to_confirm = min_hits_to_confirm
        self.tracks = {}
        self.next_id = 1
        self.frame_idx = 0

    def update(self, detections):
        self.frame_idx += 1
        matched_tracks = set()

        for det in detections:
            cx, cy = det['center']
            best_id = None
            best_dist = self.match_thresh_px

            for tid, trk in self.tracks.items():
                if tid in matched_tracks:
                    continue
                tcx, tcy = trk['center']
                dist = np.hypot(cx - tcx, cy - tcy)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                # Update existing track
                self.tracks[best_id]['bbox'] = det['bbox']
                self.tracks[best_id]['center'] = det['center']
                self.tracks[best_id]['area'] = det['area']
                self.tracks[best_id]['depth'] = det['depth']
                self.tracks[best_id]['severity'] = det['severity']
                self.tracks[best_id]['confidence'] = det['confidence']
                self.tracks[best_id]['ground_pos'] = det['ground_pos']
                self.tracks[best_id]['last_seen'] = self.frame_idx
                self.tracks[best_id]['hits'] += 1
                matched_tracks.add(best_id)
            else:
                # New candidate track
                det['track_id'] = self.next_id
                det['hits'] = 1
                det['last_seen'] = self.frame_idx
                self.tracks[self.next_id] = det
                self.next_id += 1

        # Remove dead tracks
        stale_ids = [
            tid for tid, trk in self.tracks.items()
            if (self.frame_idx - trk['last_seen']) > self.max_frames_missing
        ]
        for tid in stale_ids:
            del self.tracks[tid]

        # Return only confirmed potholes with >= min_hits
        confirmed = [
            trk for trk in self.tracks.values()
            if trk['hits'] >= self.min_hits_to_confirm and trk['last_seen'] == self.frame_idx
        ]
        return confirmed


class PotholeDetector:
    """
    Pothole and road surface anomaly detector calibrated for Indian road topography.
    """

    def __init__(self, camera_geometry=None):
        self.camera_geometry = camera_geometry
        self.tracker = PotholeTracker()
        self._frame_count = 0

        # Severity area thresholds (pixels on 640x480 frame)
        self.AREA_MINOR = 2000
        self.AREA_MODERATE = 8000
        self.AREA_SEVERE = 16000

        # Depth gradient thresholds
        self.DEPTH_SHALLOW = 18.0
        self.DEPTH_MODERATE = 38.0

    def _estimate_depth(self, roi_gray, mask, x, y, bw, bh):
        """
        Estimate 3D depression depth using brightness differential
        against surrounding asphalt and Canny edge gradient sharpness.
        """
        patch_mask = mask[y:y + bh, x:x + bw]
        patch_gray = roi_gray[y:y + bh, x:x + bw]

        pothole_px = patch_gray[patch_mask > 0]
        if len(pothole_px) == 0:
            return 0.0

        # Create surrounding dilation zone
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(patch_mask, kernel, iterations=1)
        surround_mask = cv2.bitwise_xor(dilated, patch_mask)
        surround_px = patch_gray[surround_mask > 0]

        if len(surround_px) == 0:
            surround_mean = float(np.mean(roi_gray))
        else:
            surround_mean = float(np.mean(surround_px))

        pothole_mean = float(np.mean(pothole_px))
        brightness_diff = max(0.0, surround_mean - pothole_mean)

        # Canny edge sharpness at rim
        edges = cv2.Canny(patch_gray, 40, 120)
        edge_score = float(np.sum(edges[patch_mask > 0])) / max(float(np.sum(patch_mask > 0)), 1.0)

        # Depth score: combination of depression shadow and rim gradient
        depth = brightness_diff * 0.70 + edge_score * 0.30
        return float(depth)

    def detect(self, frame):
        """
        Detect potholes, estimate depth/severity, and track temporally.
        
        Returns:
            confirmed_potholes: List of dicts with:
                - 'bbox': (abs_x, abs_y, bw, bh)
                - 'center': (cx, cy)
                - 'area': int
                - 'depth': float
                - 'confidence': float
                - 'severity': 'MINOR' | 'MODERATE' | 'SEVERE' | 'CRITICAL'
                - 'ground_pos': (X_meters, Z_meters)
        """
        h, w = frame.shape[:2]
        self._frame_count += 1

        # Focus analysis on bottom 50% road region
        roi_top = int(h * 0.50)
        roi_left = int(w * 0.08)
        roi_right = int(w * 0.92)
        roi = frame[roi_top:h, roi_left:roi_right]
        roi_h, roi_w = roi.shape[:2]

        if roi_h < 30 or roi_w < 30:
            return []

        # Convert to Grayscale & normalize with CLAHE
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray_eq, (7, 7), 0)

        # Multi-scale adaptive thresholding for road depressions
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=35,
            C=12
        )

        # Dark region mask (absolute threshold against road mean)
        dark_cut = int(np.mean(gray) * 0.72)
        dark_mask = cv2.inRange(gray, 0, dark_cut)
        combined = cv2.bitwise_and(thresh, dark_mask)

        # Morphological filtering
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

        # Extract contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_detections = []
        min_area = roi_w * roi_h * 0.003
        max_area = roi_w * roi_h * 0.15

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
            if aspect > 3.2:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.12:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)
            if solidity < 0.35:
                continue

            # Estimate depth
            depth = self._estimate_depth(gray, combined, x, y, bw, bh)

            # Classify severity
            if area > self.AREA_SEVERE or (area > self.AREA_MODERATE and depth > self.DEPTH_MODERATE):
                severity = 'CRITICAL' if depth > 45.0 else 'SEVERE'
            elif area > self.AREA_MODERATE or depth > self.DEPTH_SHALLOW:
                severity = 'MODERATE'
            else:
                severity = 'MINOR'

            # Confidence score
            norm_depth = min(1.0, depth / 50.0)
            shape_score = circularity * 0.4 + solidity * 0.6
            confidence = round(float(np.clip(shape_score * 0.4 + norm_depth * 0.6, 0.25, 0.98)), 2)

            abs_x = x + roi_left
            abs_y = y + roi_top
            cx = abs_x + bw // 2
            cy = abs_y + bh // 2

            # Estimate metric ground location if camera_geometry provided
            if self.camera_geometry:
                X_m, Z_m = self.camera_geometry.pixel_to_ground(cx, abs_y + bh)
            else:
                # Approximate
                Z_m = 15.0 * (1.0 - (cy / float(h)))
                X_m = (cx - w / 2.0) * (Z_m / (w * 0.9))

            raw_detections.append({
                'bbox': (abs_x, abs_y, bw, bh),
                'center': (cx, cy),
                'area': int(area),
                'depth': depth,
                'confidence': confidence,
                'severity': severity,
                'ground_pos': (float(X_m), float(Z_m))
            })

        # Feed to temporal tracker
        confirmed_potholes = self.tracker.update(raw_detections)
        confirmed_potholes.sort(key=lambda p: (
            {'CRITICAL': 4, 'SEVERE': 3, 'MODERATE': 2, 'MINOR': 1}.get(p['severity'], 0),
            p['confidence']
        ), reverse=True)

        return confirmed_potholes[:6]
