from ultralytics import YOLO
import cv2
import torch

# ---------------------------------------------------------------------------
#  Indian Road Class Mapping
#  Maps COCO class IDs to Indian road context names
# ---------------------------------------------------------------------------
INDIA_CLASS_NAMES = {
    0:  'Person',
    1:  'Bicycle / Cycle Rickshaw',
    2:  'Car / Auto',        # Auto-rickshaws often detected as car
    3:  'Bike / Scooter',    # motorcycle class
    5:  'Bus',
    7:  'Lorry / Truck',
    9:  'Traffic Light',
    11: 'Stop Sign',
    15: 'Cat',
    16: 'Dog',
    17: 'Horse',
    18: 'Sheep / Goat',      # COCO sheep — covers goats visually
    19: 'Cow / Bull',        # Very common on Indian roads
    20: 'Elephant',
}

# All COCO classes relevant to Indian driving
# Persons, vehicles, traffic signs, and common road animals
INDIA_ROAD_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11, 15, 16, 17, 18, 19, 20]


class ObjectDetector:
    def __init__(self, model_path='yolo11n.pt'):
        print(f"Loading YOLO model from {model_path}...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model = YOLO(model_path, task='detect')
        self.device = "unknown"
        self._is_engine = str(model_path).endswith('.engine')

    def detect(self, frame, imgsz=640, conf=0.35, iou=0.50, device=None, classes=None):
        """
        Detects objects in the frame, optimized for Indian road conditions.

        Args:
            frame: Input image (BGR)
            imgsz: Inference size (640 recommended)
            conf:  Confidence threshold.
                   0.35 gives higher recall (detects more objects including
                   partially occluded ones common in Indian traffic) while
                   still filtering pure noise.
            iou:   NMS IoU threshold — 0.50 suppresses overlapping duplicates
            device: 'cuda', 'cpu', '0', etc.
            classes: List of COCO class IDs to detect
        Returns the results object from ultralytics.
        """
        if self._is_engine:
            device = 0 if device is None else device
            self.device = "GPU (TensorRT)"
        else:
            self.device = str(device) if device else "Auto"

        if classes is None:
            classes = INDIA_ROAD_CLASSES

        results = self.model(
            frame,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            classes=classes,
            verbose=False,
            agnostic_nms=True,   # Prevents cross-class duplicates (bus+car on same object)
        )
        return results[0]

    @staticmethod
    def get_india_name(cls_id, default_name='Unknown'):
        """Get the Indian road context name for a COCO class ID."""
        return INDIA_CLASS_NAMES.get(cls_id, default_name)
