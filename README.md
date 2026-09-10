# 🚗 Fully Autonomous Driving System (ADAS) for Jetson Orin Nano

A comprehensive, real-time perception and control system designed for the NVIDIA Jetson Orin Nano. This system integrates high-speed object detection, drivable area tracking, pothole detection, and collision avoidance with hardware-level motor control via an ESP32. 

**Specialized for Indian Road Conditions:** The AI models and collision metrics are specifically tuned to handle the chaos of unlaned traffic, recognizing auto-rickshaws, cycles, cows, dogs, and factoring in the unpredictable nature of road animals.

![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Orin-green.svg)
![AI-Model](https://img.shields.io/badge/Model-YOLOv11%2B%20TensorRT-orange.svg)
![Language](https://img.shields.io/badge/Language-Python%203%20%7C%20C%2B%2B-blue.svg)

---

## 🌟 Key Features

### 🧠 Advanced Perception
- **Object Detection (YOLOv11 + TensorRT):** Detects vehicles, pedestrians, traffic lights (color-aware), stop signs, and 10+ types of animals.
- **Drivable Area Detection:** Instead of relying on painted lane lines (which are often missing), the system uses dynamic LAB color-space sampling to find the actual drivable road surface.
- **Pothole Detection:** Identifies and categorizes potholes (Minor, Moderate, Severe) to execute dodge maneuvers.

### 🛡️ Active Safety
- **Collision Avoidance:** Uses a Region of Interest (ROI) and pinhole camera focal length approximations to estimate object distances in meters.
- **Adaptive Threat Levels:** Maintains stricter distance thresholds for unpredictable animals compared to standard vehicles.
- **Driving State Machine:** Intelligently switches between `CRUISING`, `AVOID LEFT/RIGHT`, `BRAKING`, and `STOPPED_RED` based on live telemetry.

### 🔌 Hardware & Telemetry
- **ESP32 Motor Control:** Translates steering angles and throttle requests into physical PWM signals via PySerial.
- **HUD Dashboard:** A modern, full-screen OpenCV graphical interface displaying FPS, a virtual steering bar, throttle levels, and floating warning badges.
- **Streamlit Web Panel:** An alternative browser-based interface for manual override and telemetry monitoring.

---

## 🛠️ Hardware Requirements

- **Processor:** NVIDIA Jetson Orin Nano (L4T R36+)
- **Camera:** USB Webcam (optimized at 640x480 for 30 FPS)
- **Motor Control:** ESP32 connected via USB (`/dev/ttyUSB0`)
- **Power Mode:** Jetson should be set to 15W MAXN mode via `nvpmodel` for peak AI performance.

---

## 📥 Installation

1. **Clone the repository:**
   ```bash
   cd ~/Desktop
   git clone [your-repo-link]
   cd "fully autonomous driving system"
   ```

2. **Install Python Dependencies:**
   *Note: On Jetson, ensure you use the NVIDIA-provided PyTorch/Torchvision wheels to enable CUDA.*
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Model Preparation:**
   The system will automatically look for `yolo11n.engine` (TensorRT) for maximum speed. If not found, it falls back to the PyTorch `.pt` model (`yolov8n.pt`).

---

## 🚀 How to Run

### 1. USB Camera Mode (High-Performance Dashcam)
This is the primary mode, utilizing asynchronous threading for perception pipelines.

```bash
# Camera only (No ESP32 connection)
python3 autonomous_driving_system/main.py --source 0 --no-serial

# Full Autonomous Mode (Controls motors via ESP32)
python3 autonomous_driving_system/main.py --source 0 --serial-port /dev/ttyUSB0
```

### 2. Video Processing Mode
Test the AI logic on pre-recorded dashcam footage.
```bash
python3 autonomous_driving_system/main.py --source driving.mp4 --no-serial --output results.mp4
```

### 3. Streamlit Web Interface
Launch a visual control panel accessible from any browser on the network.
```bash
streamlit run app.py
```
*Access via: `http://localhost:8501`*

---

## ⚙️ Configuration Flags

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--source` | Camera index (`0`, `1`) or video file path | `0` |
| `--output` | Path to save processed video (e.g., `out.mp4`) | `None` |
| `--fix-color` | Fix camera color inversion (`swap_rg` or `bgr2rgb`) | `None` |
| `--no-serial` | Run without ESP32 motor connection | `False` |
| `--device` | "0" (GPU) or "cpu" | Auto |

---

## 📂 Project Architecture

*   `autonomous_driving_system/main.py`: The orchestrator and CLI entry point. Manages the worker threads and renders the HUD.
*   `autonomous_driving_system/perception/`: 
    *   `object_detector.py`: YOLOv11 wrapper with Indian road class mapping and Class-Agnostic NMS.
    *   `lane_detector.py`: Drivable road surface extraction using dynamic LAB color thresholding.
    *   `collision_warning.py`: ROI definition, distance estimation, and traffic light color state analysis.
    *   `pothole_detector.py`: Surface anomaly detection.
*   `autonomous_driving_system/utils/`: Hardware communication (`serial_controller.py`).
*   `app.py`: Streamlit web dashboard.
*   `deepstream_yolo/`: Configurations for deploying the AI pipeline via NVIDIA DeepStream SDK for extreme C++ throughput.
*   `esp32_motor_control/`: Firmware for the ESP32 microcontroller.

---

## 🔧 Troubleshooting

- **"Alien" Camera Colors (Red/Blue Swapped):** 
  Run `python3 debug_camera.py` to identify your color swap. Then run the main script with `--fix-color swap_rg` or `--fix-color bgr2rgb`.
- **Low FPS (< 10):** 
  Ensure you are using the `.engine` TensorRT model, not the `.pt` file. Also, verify your Jetson power mode using `sudo nvpmodel -m 0` (MAXN mode).
- **Serial Permission Denied:** 
  Run `sudo chmod 666 /dev/ttyUSB0` to grant USB serial access to the ESP32.
- **DeepStream Integration:**
  If exploring the `deepstream_yolo` folder, ensure you compile the custom bounding box parser (`nvdsinfer_custom_impl_Yolo`) against your specific DeepStream version.
