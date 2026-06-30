import cv2
import threading
import time

def test_imshow_main_thread():
    print("\n--- Testing imshow in MAIN thread ---")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera 0")
        return

    print("Camera opened. Showing window for 3 seconds...")
    start_time = time.time()
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        cv2.imshow("Main Thread Test", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()
    print("Main thread test finished.")

def show_in_thread(cap):
    print("Thread started. Showing window for 3 seconds...")
    start_time = time.time()
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame in thread")
            break
        cv2.imshow("Worker Thread Test", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    print("Worker thread finished.")

def test_imshow_worker_thread():
    print("\n--- Testing imshow in WORKER thread ---")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera 0")
        return

    t = threading.Thread(target=show_in_thread, args=(cap,))
    t.start()
    t.join()
    cap.release()
    print("Worker thread test finished.")

test_imshow_main_thread()
time.sleep(1)
test_imshow_worker_thread()
