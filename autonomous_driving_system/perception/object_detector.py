try:
    from ultralytics import YOLO
    import torch
    HAS_YOLO = True
except Exception as e:
    HAS_YOLO = False
    YOLO = None
    torch = None

import cv2

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
        self.device = "unknown"
        self._is_engine = str(model_path).endswith('.engine')

        if HAS_YOLO:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = YOLO(model_path, task='detect')
        else:
            print(f"[WARN] YOLO / PyTorch runtime unavailable in current host environment. Running fallback mode.")
            self.model = None

    def detect(self, frame, imgsz=640, conf=0.35, iou=0.50, device=None, classes=None):
        """
        Detects objects in the frame, optimized for Indian road conditions.

        Args:
            frame: Input image (BGR)
            imgsz: Inference size (640 recommended)
            conf:  Confidence threshold.
            iou:   NMS IoU threshold — 0.50 suppresses overlapping duplicates
            device: 'cuda', 'cpu', '0', etc.
            classes: List of COCO class IDs to detect
        Returns the results object from ultralytics.
        """
        if not HAS_YOLO or self.model is None:
            # Fallback mock container for environments where PyTorch isn't available
            class EmptyResults:
                boxes = []
                names = INDIA_CLASS_NAMES
            return EmptyResults()

        if self._is_engine:
            device = 0 if device is None else device
            self.device = "GPU (TensorRT)"
        else:
            if device is None or device == '':
                device = 0 if (torch and torch.cuda.is_available()) else 'cpu'
            self.device = f"PyTorch ({device})"

        classes = INDIA_ROAD_CLASSES if classes is None else classes

        results = self.model(
            frame,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            classes=classes,
            agnostic_nms=True,   # Class-Agnostic NMS: suppresses duplicate bounding boxes
            verbose=False,
            stream=False,
        )
        return results[0]

    @staticmethod
    def get_india_name(cls_id, default_name=''):
        """Convert standard COCO class name to Indian road context name."""
        return INDIA_CLASS_NAMES.get(cls_id, default_name)
