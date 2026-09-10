#!/bin/bash
# ==============================================================================
#  NVIDIA Jetson Orin Nano (8GB, JetPack 6 / L4T R36+) Setup & Optimizer Script
#  Fully Autonomous Driving System (ADAS) - Edge Deployment
# ==============================================================================

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}   NVIDIA Jetson Orin Nano - ADAS Deployment & Performance Setup ${NC}"
echo -e "${CYAN}================================================================${NC}"

# 1. System Privileges Check
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}[NOTE] Please run this script as regular user with sudo privileges, not root directly.${NC}"
fi

# 2. Maximize Hardware Clocks & Power Mode (MAXN 15W)
echo -e "\n${GREEN}[1/5] Configuring Jetson Power Profile & Hardware Clocks...${NC}"
if command -v nvpmodel &> /dev/null; then
    echo "      Setting Power Mode to MAXN (Mode 0, 15W)..."
    sudo nvpmodel -m 0 || true
    echo "      Current Power Model:"
    sudo nvpmodel -q || true
else
    echo -e "${YELLOW}      nvpmodel not found (non-Tegra host). Skipping.${NC}"
fi

if command -v jetson_clocks &> /dev/null; then
    echo "      Locking CPU, GPU, and Memory (EMC) clocks to maximum..."
    sudo jetson_clocks || true
    sudo jetson_clocks --show || true
else
    echo -e "${YELLOW}      jetson_clocks not found. Skipping.${NC}"
fi

# 3. USB-Serial Ultra-Low Latency Optimization (ESP32 Actuator)
echo -e "\n${GREEN}[2/5] Optimizing USB-Serial Latency for ESP32 Actuator...${NC}"
echo "      Reducing serial buffer latency from 16ms to 1ms for instant AEB cutoff..."

# Udev rule for non-root serial access & low latency
UDEV_RULE_PATH="/etc/udev/rules.d/99-jetson-adas-serial.rules"
echo 'KERNEL=="ttyUSB*", MODE="0666", ATTR{latency_timer}="1"' | sudo tee "$UDEV_RULE_PATH" > /dev/null
echo 'KERNEL=="ttyACM*", MODE="0666"' | sudo tee -a "$UDEV_RULE_PATH" > /dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger || true

# Apply immediately to existing USB serial devices
for dev in /sys/bus/usb-serial/devices/ttyUSB*; do
    if [ -d "$dev" ]; then
        echo 1 | sudo tee "$dev/latency_timer" > /dev/null || true
        echo "      Set latency_timer=1 on $(basename $dev)"
    fi
done

# Add user to dialout and video groups
sudo usermod -aG dialout "$USER" || true
sudo usermod -aG video "$USER" || true
echo "      Added $USER to dialout and video groups."

# 4. Python Environment & Dependencies
echo -e "\n${GREEN}[3/5] Installing/Verifying Python 3 Packages...${NC}"
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-dev python3-opencv v4l-utils

echo "      Installing Python requirements..."
pip3 install --upgrade pip
pip3 install ultralytics pyserial numpy filterpy streamlit

# 5. Export TensorRT Model for Ampere GPU
echo -e "\n${GREEN}[4/5] Building TensorRT FP16 Engine for YOLO11n...${NC}"
if [ ! -f "yolo11n.engine" ]; then
    echo "      yolo11n.engine not found. Triggering automated export..."
    python3 export_tensorrt.py --model yolo11n.pt --imgsz 480 640 --half || {
        echo -e "${YELLOW}[WARNING] Automated engine compilation failed or running on non-CUDA host.${NC}"
        echo -e "${YELLOW}PyTorch fallback (yolo11n.pt) will be used until engine is built.${NC}"
    }
else
    echo "      yolo11n.engine is already present in $(pwd)!"
fi

# 6. Verification and Ready
echo -e "\n${GREEN}[5/5] Checking Connected Video Devices & Serial Port...${NC}"
v4l2-ctl --list-devices 2>/dev/null || ls -l /dev/video* 2>/dev/null || echo "No video devices found."
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "No ESP32 serial devices plugged in currently."

echo -e "\n${CYAN}================================================================${NC}"
echo -e "${CYAN} Setup Complete! To run the Autonomous Driving System:${NC}"
echo -e "   1. Camera Mode (Real-time HUD with ESP32 Serial):"
echo -e "      ${GREEN}python3 autonomous_driving_system/main.py --source 0${NC}"
echo -e "   2. Web Control Panel (Streamlit UI):"
echo -e "      ${GREEN}streamlit run app.py${NC}"
echo -e "${CYAN}================================================================${NC}"
