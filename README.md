# 🚗 Autonomous Driving System (ADAS) for Jetson Orin Nano

A comprehensive real-time perception and control system designed for the NVIDIA Jetson Orin Nano. This system integrates high-speed object detection, lane tracking, and collision avoidance with hardware-level motor control.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Orin-green.svg)
![AI-Model](https://img.shields.io/badge/Model-YOLOv11%2B%20TensorRT-orange.svg)

---

## 🌟 Key Features

- **🚀 High Performance**: Powered by YOLOv11 with TensorRT acceleration on Jetson GPU.
- **🛡️ Multi-Object Detection**: Detects vehicles, pedestrians, traffic lights (color-aware), stop signs, and 10+ types of animals.
- **🛣️ Lane Keeping**: Real-time Hough Transform-based lane detection with a software P-controller for steering.
- **⚠️ Collision Avoidance**: Intelligent ROI-based danger assessment with automatic throttle cutoff and "Critical Alert" warnings.
- **🔌 Hardware Control**: Direct serial communication with ESP32 for Motor/Servo control.
- **🌐 Dual Interface**: Choose between a high-speed CLI Dashcam mode or an interactive Streamlit Web Panel.

---

## 🛠️ Hardware Requirements

- **Processor**: NVIDIA Jetson Orin Nano (L4T R36+)
- **Camera**: USB Webcam or CSI Camera (configured for 640x480 for optimal FPS)
- **Motor Control**: ESP32 connected via USB (optional for AI-driven steering)
- **Display**: HDMI monitor for live visualization

---

## 📥 Installation

1. **Clone the project**
   ```bash
   cd ~/Desktop
   git clone [your-repo-link]
   cd "fully autonomous driving system"
   ```

2. **Setup Dependencies** (Ensure you have Jetson-optimized PyTorch/TensorRT installed)
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Model Preparation**
   The system automatically looks for `yolo11n.engine` on the Desktop. If not found, it falls back to `.pt` weights or downloads them.

---

## 🚀 How to Run

### 1. Dashcam Mode (Primary)
Optimized for maximum FPS with multi-threaded processing and fullscreen display.

```bash
# Camera only (No ESP32)
python3 autonomous_driving_system/main.py --source 0 --no-serial

# Full Autonomous Mode (Camera + ESP32 Control)
python3 autonomous_driving_system/main.py --source 0 --serial-port /dev/ttyUSB0

# Fix color if red/green are swapped
python3 autonomous_driving_system/main.py --source 0 --fix-color swap_rg
```

### 2. Video Processing Mode
Test the AI logic on pre-recorded dashcam footage.

```bash
python3 autonomous_driving_system/main.py --source driving.mp4 --no-serial --output results.mp4
```

### 3. Streamlit Web Interface
Visual control panel with live telemetry and manual override sliders.

```bash
streamlit run app.py
```
*Access via: `http://localhost:8501`*

---

## ⚙️ Configuration (Command Line Options)

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--source` | Camera index (0, 1) or video file path | `0` |
| `--output` | Path to save processed video (e.g., `out.mp4`) | `None` |
| `--fix-color` | Fix camera color inversion (`swap_rg` or `bgr2rgb`) | `None` |
| `--no-serial` | Run without ESP32 motor connection | `False` |
| `--device` | "0" (GPU) or "cpu" | `0` |

---

## 📂 Project Structure

- `autonomous_driving_system/main.py`: Entry point for the CLI system.
- `autonomous_driving_system/perception/`: AI logic (Objects, Lanes, Collision).
- `autonomous_driving_system/utils/`: Hardware communication drivers.
- `app.py`: Streamlit web dashboard.
- `HOW_TO_RUN.txt`: Legacy quick-start notes.

---

## 🔧 Troubleshooting

- **Low FPS**: Ensure `yolo11n.engine` is present. TensorRT is ~5x faster than PyTorch on Jetson.
- **Serial Denied**: Run `sudo chmod 666 /dev/ttyUSB0`.
- **Camera Error**: Check permissions or try `--source 1`.
- **Color Issues**: Use `--fix-color swap_rg` if red/green channels are inverted by the driver.

---
*Created for the Fully Autonomous Driving System Project.*
