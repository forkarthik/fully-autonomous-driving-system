"""
Camera Geometry and Inverse Perspective Mapping (IPM)
Calibrated for NVIDIA Jetson Orin Nano forward-facing ADAS camera.

Transforms 2D image coordinates (u, v) into real-world 3D/ground-plane
coordinates (X_meters, Z_meters) relative to vehicle front bumper.
Inspired by OpenADAS and Swaayatt Robots ground-plane projection models.
"""

import numpy as np
import cv2


class CameraGeometry:
    """
    Ground-plane geometry and metric distance estimator.
    
    Coordinate System (ISO / Vehicle frame):
      - Z: Forward distance in meters ahead of front bumper
      - X: Lateral distance in meters (Right = positive, Left = negative)
      - Y: Vertical elevation (Down = positive)
    """

    def __init__(self, frame_width=640, frame_height=480, camera_height=1.2, camera_pitch_deg=4.0):
        self.w = frame_width
        self.h = frame_height
        
        # Camera physical mounting parameters
        self.cam_height = camera_height  # Height above road in meters (e.g. 1.2m on hood/windshield)
        self.pitch_rad = np.radians(camera_pitch_deg)  # Slight downward tilt
        
        # Intrinsic focal length approximation for standard 640x480 webcam (~60-70 deg H-FOV)
        self.fx = self.w * 0.95
        self.fy = self.w * 0.95
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0

        # Precompute vanishing line horizon y-coordinate
        # Horizon occurs at cy - fy * tan(pitch)
        self.horizon_y = int(self.cy - self.fy * np.tan(self.pitch_rad))
        self.horizon_y = max(int(self.h * 0.35), min(int(self.h * 0.55), self.horizon_y))

        # Real-world reference widths (meters) for Indian vehicles & common obstacles
        self.real_widths = {
            0: 0.50,   # Pedestrian / person
            1: 0.60,   # Bicycle / Cycle-rickshaw
            2: 1.65,   # Car / Auto-rickshaw (smaller than western sedans)
            3: 0.70,   # Motorcycle / Scooter
            5: 2.50,   # Bus
            7: 2.50,   # Lorry / Truck
            15: 0.35,  # Cat
            16: 0.45,  # Dog
            17: 1.50,  # Horse
            18: 0.55,  # Sheep / Goat
            19: 1.80,  # Cow / Bull
            20: 3.00,  # Elephant
            'default': 1.0,
        }

    def pixel_to_ground(self, u, v):
        """
        Projects image pixel (u, v) onto flat ground plane (Y = cam_height).
        
        Returns:
            (X_meters, Z_meters)
            Returns (0.0, 100.0) if point is above or near horizon.
        """
        if v <= self.horizon_y + 5:
            return 0.0, 100.0  # At or above horizon -> distant / infinity

        # Normalized camera ray vector
        ray_x = (u - self.cx) / self.fx
        ray_y = (v - self.cy) / self.fy
        ray_z = 1.0

        # Rotate ray by camera pitch angle around X axis:
        pitch_cos = np.cos(self.pitch_rad)
        pitch_sin = np.sin(self.pitch_rad)
        
        rotated_y = ray_y * pitch_cos + ray_z * pitch_sin
        rotated_z = -ray_y * pitch_sin + ray_z * pitch_cos
        rotated_x = ray_x

        if rotated_y <= 0.001:
            return 0.0, 100.0

        # Scale ray to hit ground plane at Y = cam_height
        scale = self.cam_height / rotated_y
        Z = rotated_z * scale
        X = rotated_x * scale

        # Clamp physically reasonable range (0.5m to 100m)
        Z = float(np.clip(Z, 0.5, 100.0))
        X = float(np.clip(X, -25.0, 25.0))
        return X, Z

    def ground_to_pixel(self, X, Z):
        """
        Reverse projection: maps ground coordinate (X, Z in meters) back to image pixel (u, v).
        """
        if Z <= 0.2:
            return int(self.cx), self.h - 1

        Y = self.cam_height
        pitch_cos = np.cos(-self.pitch_rad)
        pitch_sin = np.sin(-self.pitch_rad)

        cam_y = Y * pitch_cos + Z * pitch_sin
        cam_z = -Y * pitch_sin + Z * pitch_cos
        cam_x = X

        if cam_z <= 0.05:
            return int(self.cx), self.h - 1

        u = int(self.cx + self.fx * (cam_x / cam_z))
        v = int(self.cy + self.fy * (cam_y / cam_z))
        return u, v

    def estimate_object_position(self, bbox, cls_id=None):
        """
        Estimate 3D ground location (X, Z) and distance for a bounding box.
        Fuses ground-contact touchpoint (bottom center of bbox) with bounding-box pinhole width.
        
        Args:
            bbox: (x1, y1, x2, y2)
            cls_id: optional COCO class ID for width validation
        
        Returns:
            X: lateral offset (meters, 0=center, +right, -left)
            Z: forward distance (meters)
        """
        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1)
        bottom_cx = (x1 + x2) / 2.0
        bottom_cy = float(y2)

        # 1. Ground contact projection
        X_ground, Z_ground = self.pixel_to_ground(bottom_cx, bottom_cy)

        # 2. Pinhole model from bounding box width
        w_real = self.real_widths.get(cls_id, self.real_widths['default']) if cls_id is not None else 1.0
        Z_pinhole = (w_real * self.fx) / box_w

        # Fusion: ground projection is best near-to-mid range; width model validates against occlusion
        if Z_ground < 35.0:
            Z = 0.75 * Z_ground + 0.25 * Z_pinhole
        else:
            Z = 0.40 * Z_ground + 0.60 * Z_pinhole

        X = (bottom_cx - self.cx) * Z / self.fx
        return float(X), float(Z)
