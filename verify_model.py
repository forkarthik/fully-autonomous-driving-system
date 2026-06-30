from ultralytics import YOLO
import sys
import os

engine_path = '/home/smvec-eee/Desktop/yolo11n.engine'

try:
    print(f"Attempting to load YOLO engine from {engine_path}...")
    if not os.path.exists(engine_path):
        print(f"Error: Engine file not found at {engine_path}")
        sys.exit(1)
        
    model = YOLO(engine_path)
    print("Model loaded successfully!")
    
    # Run a dummy inference to be sure
    # Create a dummy image
    import numpy as np
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(img)
    print("Inference successful!")
    
except Exception as e:
    print(f"FAILED to load model: {e}")
    sys.exit(1)
