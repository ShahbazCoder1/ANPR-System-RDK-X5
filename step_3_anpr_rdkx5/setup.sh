#!/bin/bash
set -e

echo "=========================================================="
echo "    RDK X5 ANPR Toll Booth System — Board Environment Setup"
echo "=========================================================="

# Create necessary directories
mkdir -p data/plate_crops
mkdir -p models

echo ""
echo "[Step 1/3] Installing system libraries (libssl1.1 for ARM64 PaddleOCR)..."
if ! dpkg -l | grep -q "libssl1.1"; then
    echo "Installing libssl1.1 for ARM64..."
    wget -q http://ports.ubuntu.com/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_arm64.deb
    sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2_arm64.deb || true
    rm -f libssl1.1_1.1.1f-1ubuntu2_arm64.deb
    echo "[OK] libssl1.1 installed."
else
    echo "[OK] libssl1.1 is already installed."
fi

echo ""
echo "[Step 2/3] Installing Python dependencies on RDK X5..."
pip3 install -r requirements.txt

echo ""
echo "[Step 3/3] Checking model files..."
if [ ! -f "models/yolov8n_plate_bayese_640x640_nv12.bin" ]; then
    echo "[NOTICE] Compiled BPU model not found in models/ directory."
    echo "Please copy 'yolov8n_plate_bayese_640x640_nv12.bin' into the models/ folder."
else
    echo "[OK] BPU model found: models/yolov8n_plate_bayese_640x640_nv12.bin"
fi

echo ""
echo "=========================================================="
echo " [SUCCESS] Setup completed!"
echo ""
echo " To run with a test video:"
echo "   python3 main.py --source test_video.mp4"
echo ""
echo " To run with GS130W MIPI camera:"
echo "   python3 main.py --source mipi"
echo ""
echo " Open dashboard from any browser on the same network at:"
echo "   http://<rdk-x5-ip>:5000"
echo "=========================================================="
