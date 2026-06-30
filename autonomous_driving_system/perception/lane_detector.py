import cv2
import numpy as np


class LaneDetector:
    """
    Road Surface Detector for Indian roads (laned AND unlaned).

    Instead of detecting lane LINES (which don't exist on most Indian roads),
    this detects the DRIVABLE ROAD SURFACE and creates a green overlay mask
    — exactly like the image reference.

    Algorithm:
      1. Sample road color from the bottom-center of the frame (always road)
      2. Find all pixels matching that road color in LAB color space
      3. Apply a trapezoidal ROI mask (road perspective shape)
      4. Morphological cleanup to remove noise
      5. Find the largest contour = drivable area
      6. Also detect lane lines if they exist (bonus for well-marked roads)
      7. Extract left/right boundaries for steering

    Output:
      - road_mask: filled polygon of drivable area (for green overlay)
      - lane_lines: list of line segments (if lane markings exist)
      - lanes_info: left_x, right_x, lane_center_x for steering
    """

    def __init__(self):
        # Temporal smoothing for road boundary
        self._prev_road_contour = None
        self._prev_left_x = None
        self._prev_right_x = None
        self._smooth_alpha = 0.3

        # Road color reference (will be sampled from frame)
        self._road_color_lab = None
        self._color_update_count = 0

    def _sample_road_color(self, frame):
        """
        Sample the road color from the bottom-center strip of the frame.
        This region is almost always the road surface directly ahead.
        """
        h, w = frame.shape[:2]
        # Sample a horizontal strip at 90% height, center 40% width
        y_start = int(h * 0.85)
        y_end = h
        x_start = int(w * 0.30)
        x_end = int(w * 0.70)

        roi = frame[y_start:y_end, x_start:x_end]
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        # Use median color (robust to small objects on road)
        median_color = np.median(lab_roi.reshape(-1, 3), axis=0)
        return median_color

    def _detect_road_surface(self, frame):
        """
        Detect the drivable road surface using color similarity
        to the sampled road color in LAB color space.
        """
        h, w = frame.shape[:2]

        # Update road color reference periodically (every 10 frames)
        self._color_update_count += 1
        if self._road_color_lab is None or self._color_update_count % 10 == 0:
            self._road_color_lab = self._sample_road_color(frame)

        # Convert frame to LAB color space (perceptually uniform)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref = self._road_color_lab.astype(np.float32)

        # Calculate color distance from road reference
        diff = lab - ref
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        # Threshold: pixels within color distance are "road"
        # Adaptive threshold based on road color consistency
        threshold = 35.0  # LAB distance units
        road_mask = (dist < threshold).astype(np.uint8) * 255

        # Apply perspective ROI (road is trapezoidal)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_polygon = np.array([[
            (int(w * 0.0), h),           # Bottom-left
            (int(w * 1.0), h),           # Bottom-right
            (int(w * 0.65), int(h * 0.40)),  # Top-right
            (int(w * 0.35), int(h * 0.40)),  # Top-left
        ]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_polygon, 255)
        road_mask = cv2.bitwise_and(road_mask, roi_mask)

        # Morphological cleanup
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_close)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_open)

        # Fill holes
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

        return road_mask

    def _extract_road_contour(self, road_mask, frame_width, frame_height):
        """
        Find the largest contour in the road mask = the drivable area.
        Returns the contour points and left/right boundary x-coordinates.
        """
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, None

        # Largest contour = main road area
        road_contour = max(contours, key=cv2.contourArea)

        # Minimum area check (at least 5% of frame)
        min_area = frame_width * frame_height * 0.05
        if cv2.contourArea(road_contour) < min_area:
            return None, None, None

        # Extract left and right boundaries at a reference y-level
        # Use 85% of frame height (near bottom, where we steer)
        ref_y = int(frame_height * 0.85)
        left_x = None
        right_x = None

        # Scan the road mask at ref_y to find leftmost and rightmost road pixels
        row = road_mask[ref_y, :]
        road_pixels = np.where(row > 0)[0]
        if len(road_pixels) > 10:
            left_x = int(road_pixels[0])
            right_x = int(road_pixels[-1])

        return road_contour, left_x, right_x

    def _detect_lane_lines(self, frame):
        """
        Optional: detect actual lane markings if they exist.
        Returns list of line segments for overlay.
        """
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        median_val = np.median(blur)
        edges = cv2.Canny(blur, int(max(30, 0.6 * median_val)),
                          int(min(255, 1.4 * median_val)))

        # ROI mask
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.95), h),
            (int(w * 0.60), int(h * 0.50)),
            (int(w * 0.40), int(h * 0.50)),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked, 2, np.pi / 180, 50,
                                minLineLength=50, maxLineGap=100)

        result = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.4 or abs(slope) > 3.5:
                    continue
                result.append(((x1, y1), (x2, y2)))

        return result

    def _smooth_value(self, new_val, prev_val):
        """Temporal smoothing for a single value."""
        if prev_val is None:
            return new_val
        if new_val is None:
            return prev_val
        a = self._smooth_alpha
        return int(a * new_val + (1 - a) * prev_val)

    def detect(self, frame):
        """
        Detect the drivable road surface and optional lane lines.

        Returns:
            result: dict with:
                'road_contour': np.array of road boundary points (for green fill)
                'road_mask': binary mask of drivable area
                'lane_lines': list of ((x1,y1),(x2,y2)) line segments
                'left_x': left road boundary x at bottom
                'right_x': right road boundary x at bottom
                'lane_center_x': center of drivable area
                'lane_width': width of drivable area
        """
        height, width = frame.shape[:2]

        # 1. Detect road surface
        road_mask = self._detect_road_surface(frame)

        # 2. Extract road contour and boundaries
        road_contour, left_x, right_x = self._extract_road_contour(
            road_mask, width, height
        )

        # 3. Temporal smoothing on boundaries
        left_x = self._smooth_value(left_x, self._prev_left_x)
        right_x = self._smooth_value(right_x, self._prev_right_x)
        self._prev_left_x = left_x
        self._prev_right_x = right_x

        if road_contour is not None:
            self._prev_road_contour = road_contour

        # 4. Optional lane line detection
        lane_lines = self._detect_lane_lines(frame)

        # 5. Build output
        lanes_info = {
            'left_x': left_x,
            'right_x': right_x,
            'lane_center_x': None,
            'lane_width': None,
        }
        if left_x is not None and right_x is not None:
            lanes_info['lane_center_x'] = int((left_x + right_x) / 2)
            lanes_info['lane_width'] = int(abs(right_x - left_x))

        result = {
            'road_contour': self._prev_road_contour,
            'road_mask': road_mask,
            'lane_lines': lane_lines,
            'lanes_info': lanes_info,
        }

        return result
