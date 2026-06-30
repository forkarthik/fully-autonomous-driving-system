import cv2
import numpy as np


class PotholeDetector:
    """
    Pothole detector for Indian roads using computer vision.

    Detects potholes as dark, roughly circular depressions on the road surface.
    Draws green bounding boxes with confidence scores — similar to the
    NVIDIA Smart Pothole Detector reference.

    Algorithm:
      1. Extract road ROI (bottom 45% of frame, center 80% width)
      2. Multi-channel analysis (Grayscale + Saturation)
      3. Adaptive thresholding to find dark patches
      4. Morphological cleanup
      5. Contour filtering (area, aspect ratio, circularity, solidity)
      6. Confidence scoring based on feature strength
    """

    def __init__(self):
        self._frame_count = 0

    def detect(self, frame):
        """
        Detect potholes in the frame.

        Returns:
            potholes: List of dicts:
                'bbox': (x, y, w, h) in original frame coords
                'center': (cx, cy)
                'confidence': float 0.0–1.0
                'severity': 'MINOR' | 'MODERATE' | 'SEVERE'
        """
        h, w = frame.shape[:2]
        self._frame_count += 1

        # --- 1. Extract road ROI ---
        roi_top = int(h * 0.55)
        roi_left = int(w * 0.10)
        roi_right = int(w * 0.90)
        roi = frame[roi_top:h, roi_left:roi_right]
        roi_h, roi_w = roi.shape[:2]

        if roi_h < 30 or roi_w < 30:
            return []

        # --- 2. Multi-channel preprocessing ---
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # CLAHE for contrast normalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # Also use saturation channel (potholes often have different texture)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]

        blur = cv2.GaussianBlur(gray_eq, (7, 7), 0)

        # --- 3. Adaptive thresholding ---
        # Potholes are DARKER than surrounding road
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=10
        )

        # Also threshold on low saturation (wet potholes can be shiny)
        # and very dark absolute values
        dark_mask = cv2.inRange(gray, 0, int(np.mean(gray) * 0.65))

        # Combine: must be dark AND locally darker than surroundings
        combined = cv2.bitwise_and(thresh, dark_mask)

        # --- 4. Morphological cleanup ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # --- 5. Find and filter contours ---
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        potholes = []
        min_area = roi_w * roi_h * 0.003   # At least 0.3% of ROI
        max_area = roi_w * roi_h * 0.12    # No more than 12% of ROI

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # Aspect ratio: potholes are roughly round/oval (not elongated)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
            if aspect > 3.5:
                continue

            # Circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.12:
                continue

            # Solidity (ratio of contour area to convex hull area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1:
                continue
            solidity = area / hull_area
            if solidity < 0.3:
                continue

            # --- 6. Confidence scoring ---
            # Higher confidence if:
            #   - More circular
            #   - Higher solidity
            #   - Darker than surroundings
            #   - Good aspect ratio

            # Darkness score: how much darker is the patch vs. surroundings?
            patch_mean = np.mean(gray[y:y + bh, x:x + bw])
            surround_mean = np.mean(gray)
            darkness_score = max(0, min(1, (surround_mean - patch_mean) / (surround_mean + 1e-5)))

            # Shape score
            shape_score = (circularity * 0.4 + solidity * 0.3 +
                           max(0, 1.0 - abs(aspect - 1.0) * 0.3) * 0.3)

            # Area score (medium-sized patches are more confident)
            area_ratio = area / (roi_w * roi_h)
            area_score = min(1.0, area_ratio * 30)

            confidence = (darkness_score * 0.4 + shape_score * 0.35 + area_score * 0.25)
            confidence = round(min(0.99, max(0.10, confidence)), 2)

            # Minimum confidence filter
            if confidence < 0.25:
                continue

            # Convert to frame coordinates
            abs_x = x + roi_left
            abs_y = y + roi_top
            cx = abs_x + bw // 2
            cy = abs_y + bh // 2

            # Severity
            if area_ratio > 0.025 or confidence > 0.7:
                severity = 'SEVERE'
            elif area_ratio > 0.008 or confidence > 0.45:
                severity = 'MODERATE'
            else:
                severity = 'MINOR'

            potholes.append({
                'bbox': (abs_x, abs_y, bw, bh),
                'center': (cx, cy),
                'confidence': confidence,
                'severity': severity,
            })

        # Sort by confidence (highest first)
        potholes.sort(key=lambda p: p['confidence'], reverse=True)

        # Keep top 8
        return potholes[:8]
