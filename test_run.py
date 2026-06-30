import cv2
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.abspath('autonomous_driving_system'))

from autonomous_driving_system.perception.object_detector import ObjectDetector
from autonomous_driving_system.perception.lane_detector import LaneDetector
from autonomous_driving_system.main import draw_results

def test_system():
    print("Testing Autonomous Driving System modules...")
    
    # Create a dummy frame (black image)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw some dummy lines for lane detection
    cv2.line(frame, (100, 480), (300, 300), (255, 255, 255), 5)
    cv2.line(frame, (540, 480), (340, 300), (255, 255, 255), 5)

    # Test Object Detector
    try:
        print("Initializing ObjectDetector...")
        detector = ObjectDetector(model_path='yolov8n.pt') 
        # Mocking detection result structure is hard without running the actual model, 
        # so we will just run the detect method
        print("Running detection on dummy frame...")
        detections = detector.detect(frame)
        print("Object detection successful.")
    except Exception as e:
        print(f"Object Detection Failed: {e}")
        # Depending on environment, downloading model might fail or take time.
        # But we want to ensure the code interface is correct.

    # Test Lane Detector
    try:
        print("Initializing LaneDetector...")
        lane_detector = LaneDetector()
        print("Running lane detection...")
        lane_image, lanes_info = lane_detector.detect(frame)
        assert lane_image.shape == frame.shape
        print("Lane detection successful.")
    except Exception as e:
        print(f"Lane Detection Failed: {e}")

    # Test Visualization (only if previous steps worked, but we try anyway)
    try:
        # Need real detections object for draw_results, if detection failed this will fail
        if 'detections' in locals() and 'lane_image' in locals():
            print("Testing visualization...")
            # We don't have collision_warning in test, so we pass None or create a mock
            # But None is handled by default
            final_frame = draw_results(frame, detections, lane_image, lanes_info)
            assert final_frame.shape == frame.shape
            print("Visualization successful.")
        else:
            print("Skipping visualization test due to previous failures.")
    except Exception as e:
        print(f"Visualization Failed: {e}")

if __name__ == "__main__":
    test_system()
