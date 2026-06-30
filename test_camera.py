import cv2
import time

def test_camera(source):
    print(f"Testing camera source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open source {source}")
        return False
    
    print(f"Source {source} opened. Reading frame...")
    ret, frame = cap.read()
    if ret:
        print(f"Success! Frame shape: {frame.shape}")
        cv2.imwrite(f"test_frame_{source}.jpg", frame)
    else:
        print("Failed to read frame (ret is False)")
    
    cap.release()
    return ret

print("OpenCV build info:")
print(cv2.getBuildInformation())

# Test index 0
print("\n--- Testing VideoCapture(0) ---")
test_camera(0)

# Test index 1 (sometimes USB camera is 1 if 0 is taken)
print("\n--- Testing VideoCapture(1) ---")
test_camera(1)

# Test GStreamer pipeline (imx219 standard)
print("\n--- Testing GStreamer (nvarguscamerasrc) ---")
gst_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
test_camera(gst_str)
