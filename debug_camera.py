import cv2
import time

def main():
    print("Opening camera... Press 'q' to quit.")
    print("Controls:")
    print(" 1: Normal (Default)")
    print(" 2: BGR to RGB (Fix Blue/Red swap typically)")
    print(" 3: Swap Red/Green (Custom)")
    print(" 4: Swap Blue/Red (Manual swap)")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera 0.")
        return

    mode = 1
    mode_names = {
        1: "Normal (No Change)",
        2: "BGR to RGB (cv2.cvtColor)",
        3: "Swap Red/Green Channels",
        4: "Swap Blue/Red Channels"
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Apply color fix based on mode
            if mode == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif mode == 3:
                # Swap Red (2) and Green (1)
                frame[:, :, [1, 2]] = frame[:, :, [2, 1]]
            elif mode == 4:
                # Swap Blue (0) and Red (2)
                frame[:, :, [0, 2]] = frame[:, :, [2, 0]]

            # functional overhead for display
            cv2.putText(frame, f"Mode {mode}: {mode_names[mode]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press 1-4 to change, q to quit", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Debug Camera', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                mode = 1
                print(f"Switched to {mode_names[mode]}")
            elif key == ord('2'):
                mode = 2
                print(f"Switched to {mode_names[mode]}")
            elif key == ord('3'):
                mode = 3
                print(f"Switched to {mode_names[mode]}")
            elif key == ord('4'):
                mode = 4
                print(f"Switched to {mode_names[mode]}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
