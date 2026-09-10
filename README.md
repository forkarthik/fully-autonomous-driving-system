# 🚗 Autonomous Driving System (ADAS) for NVIDIA Jetson Orin Nano

A state-of-the-art real-time perception, planning, and control suite specifically engineered for the **NVIDIA Jetson Orin Nano 8GB (JetPack 6 / L4T R36+, ARM64)**.

Specialized for **unlaned, stochastic, and adversarial Indian road conditions** — handling chaotic traffic, darting pedestrians, stray cattle, unlaned multi-agent corridors, and severe road surface anomalies.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Orin%20Nano%20(8GB)-green.svg)
![OS](https://img.shields.io/badge/OS-JetPack%206%20%7C%20L4T%20R36%2B-blue.svg)
![AI-Model](https://img.shields.io/badge/Model-YOLOv11%2B%20TensorRT-orange.svg)
![Control](https://img.shields.io/badge/Control-Stanley%20%2B%20Follow--the--Gap-red.svg)

---

## 🌟 Synthesized Architecture & Key Capabilities

This system combines the best pros from industry and research architectures (**Autoware.Universe / Tier4**, **F1TENTH / Autonomous-Racing**, **OpenADAS**, **Pothole-detection**, and **Swaayatt Robots**):

### 1. 🧠 Multi-Object Perception & Ground Projection
- **YOLOv11 + TensorRT Engine**: Ultra-low latency inference on the Orin Nano Ampere GPU at 30+ FPS (`yolo11n.engine`).
- **Indian Road Context**: Custom mapping for auto-rickshaws, cycles, buses, lorries, cows, dogs, goats, and pedestrians.
- **Inverse Perspective Mapping (IPM)**: Calibrated flat-world homography estimating real-world ground coordinates $(X, Z)$ in meters relative to the vehicle's front bumper.

### 2. ⚡ Dynamic iTTC & Automatic Emergency Braking (AEB)
- **Instantaneous Time-To-Collision (iTTC)** (from *Autonomous-Racing*):
  $$\text{iTTC} = \frac{Z}{\max(v_{rel, z}, 0.05)}$$
- **Multi-Object Centroid Tracker**: Continuously computes relative closing rates ($v_{rel, z}$) and lateral speeds ($v_{rel, x}$).
- **Anticipatory Crossing Interception**: Detects pedestrians or cattle on the road shoulder moving towards the vehicle's travel corridor, triggering proactive yield maneuvers **before** they enter the immediate path.

### 3. 🛣️ Topographical Pothole Depth & Severity Analysis
- **Contrast & Canny Gradient Depth Estimation**: Evaluates local brightness differentials against surrounding asphalt and rim edge sharpness to estimate physical hole depth, eliminating shadow false positives.
- **Temporal Consistency Tracker**: Requires $\ge 2$ consecutive frame hits to confirm a hazard.
- **Severity Classification**: `MINOR`, `MODERATE`, `SEVERE`, and `CRITICAL` craters.

### 4. 🧭 Follow-The-Gap Free Space Planning & Pothole Dodger
- **Disparity Extender & Dynamic Safety Bubbles** (from *F1TENTH / Swaayatt Robots*):
  - Applies class-specific safety bubbles ($1.8\text{m}$ for cows, $1.2\text{m}$ for pedestrians/dogs, $1.3\text{m}$ for auto-rickshaws).
  - Scans angular free space and finds the widest, deepest safe corridor.
  - Dynamically calculates verified asphalt-safe detour waypoints around potholes.

### 5. 🎯 Stanley Lateral Controller & Tier4 Velocity Profiling
- **Stanley Control Law**:
  $$\delta = \psi + \arctan\left(\frac{k_E \cdot e_y}{v + k_v}\right)$$
- **Anti-Jitter Filtering**: Slew-rate limiting and exponential smoothing tailored for smooth ESP32 servo control.
- **Curvature-Adaptive Velocity Profiler**: Smoothly curves throttle during evasive maneuvers and executes instantaneous brake cutoff on AEB.

### 6. 🖥️ 8-Layer Modern HUD & Bird's-Eye View (BEV) Mini-Radar
1. **Layer 1 (3D Trajectory Ribbon & Rollouts)**: Holographic curving perspective trajectory ribbon with animated flowing chevrons and Swaayatt Robots candidate motion primitive rollouts.
2. **Layer 2 (Safety ROI Polygon)**: Dynamic collision corridor overlay.
3. **Layer 3 (3D Ground Footprints & Lock-On Crosshairs)**: Displays class, distance in meters, relative velocity vector ($v_{rel}$), and live iTTC countdowns.
4. **Layer 4 (Topographical Pothole Sonar Rings)**: Color-coded severity sonar rings with estimated physical depth tags (e.g., `-8.5cm [MODERATE]`).
5. **Layer 5 (Floating Threat Alerts)**: Prioritized stacked alerts for AEB, pedestrian crossing, and signal states.
6. **Layer 6 (State Badge)**: Floating pill displaying active behavioral state (`CRUISING`, `YIELD CROSSING`, `AVOID LEFT/RIGHT`, `DODGE POTHOLE`, `EMERGENCY AEB`, `RED LIGHT`).
7. **Layer 7 (Glassmorphism Cockpit Dashboard)**: Bottom instrument cluster with speedometer in KM/H, steering compass dial, throttle power bar, and Jetson MAXN 15W vitals.
8. **Layer 8 (Tactical Circular BEV Radar)**: Top-down 140x140 tactical mini-radar showing ego-vehicle, safe corridor clearance fan, obstacle positions with velocity tails, and pothole markers.

---

## 🛠️ Hardware Requirements & Jetson Configuration

- **Processor**: NVIDIA Jetson Orin Nano (8GB)
- **OS**: JetPack 6.x / L4T R36+ (Ubuntu 22.04 LTS, ARM64)
- **Camera**: USB Webcam or CSI Camera (configured for 640×480 @ 30 FPS)
- **Microcontroller**: ESP32 connected via USB (`/dev/ttyUSB0`)
- **Power Mode**: Set to 15W MAXN mode for peak GPU performance:
  ```bash
  sudo nvpmodel -m 0
  sudo jetson_clocks
  ```

---

## 📥 1-Click Jetson Deployment & Setup

We provide an automated setup and latency optimizer script tailored for JetPack 6:

```bash
# 1. Clone repository
git clone https://github.com/forkarthik/fully-autonomous-driving-system.git ADAS
cd ADAS

# 2. Run automated optimizer (sets MAXN 15W, jetson_clocks, 1ms serial latency timer, permissions)
chmod +x jetson_setup.sh
./jetson_setup.sh

# 3. (Optional) Export custom TensorRT FP16 engine with automated benchmark
python3 export_tensorrt.py --model yolo11n.pt --imgsz 480 640 --half
```

---

## 🚀 How to Run

### 1. High-Performance Dashcam Mode (Primary CLI)
```bash
# Camera only (No ESP32 connection)
python3 autonomous_driving_system/main.py --source 0 --no-serial

# Full Autonomous Driving Mode (Camera + ESP32 Motor Control with 1ms latency)
python3 autonomous_driving_system/main.py --source 0 --serial-port /dev/ttyUSB0

# If camera colors are swapped (common Linux webcam driver issue)
python3 autonomous_driving_system/main.py --source 0 --fix-color swap_rg
```

### 2. Pre-Recorded Dashcam Video Verification
```bash
python3 autonomous_driving_system/main.py --source test_video.mp4 --no-serial --output out.mp4
```

### 3. Streamlit Telemetry Web Dashboard
```bash
streamlit run app.py
```
*Access via browser: `http://localhost:8501`*

### 4. Run Integration Test Suite
```bash
python3 test_adas_integration.py
```

---

## 📂 Project Architecture

```
ADAS/
├── autonomous_driving_system/
│   ├── main.py                     # Main orchestrator, threads & next-gen Cyberpunk HUD
│   ├── perception/
│   │   ├── camera_geometry.py      # IPM & ground-plane 3D distance projection
│   │   ├── tracking_and_ttc.py     # Multi-object tracker, velocity & iTTC AEB
│   │   ├── object_detector.py      # YOLOv11 TensorRT detector with Indian classes
│   │   ├── lane_detector.py        # LAB color-space drivable road surface extractor
│   │   ├── pothole_detector.py     # Depth gradient & temporal pothole analyzer
│   │   └── collision_warning.py    # Multi-level threat & traffic light evaluator
│   ├── planning/
│   │   └── costmap_planner.py      # Follow-The-Gap corridor finder & Pothole Dodger
│   ├── control/
│   │   └── stanley_controller.py   # Stanley lateral control & Tier4 velocity profiler
│   └── utils/
│       └── serial_controller.py    # PySerial driver for ESP32 with 3-param alarm protocol
├── esp32_motor_control/            # Firmware for ESP32 microcontroller with active buzzer alarm
├── export_tensorrt.py              # Automated YOLO to TensorRT engine exporter
├── jetson_setup.sh                 # 1-click Jetson MAXN clocks & 1ms latency optimizer
├── test_adas_integration.py        # Automated test suite (100% pass rate)
├── render_demo_frame.py            # Headless HUD demonstration frame generator
├── app.py                          # Streamlit web control panel & radar display
├── debug_camera.py                 # Camera color calibration utility
└── requirements.txt
```

---

## 🔧 Troubleshooting

- **Low FPS (< 15)**: Verify `yolo11n.engine` exists. TensorRT runs ~5x faster than `.pt` models on Orin Nano. Set MAXN mode via `sudo nvpmodel -m 0`.
- **ESP32 Serial Permission Denied**: Run `sudo chmod 666 /dev/ttyUSB0` or run `./jetson_setup.sh`.
- **Inverted Camera Colors**: Run `python3 debug_camera.py` to identify your color swap, then start with `--fix-color swap_rg` or `--fix-color bgr2rgb`.

---
*Developed for the Fully Autonomous Driving System Project.*
