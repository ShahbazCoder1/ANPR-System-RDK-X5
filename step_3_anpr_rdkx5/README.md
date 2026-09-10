# Step 3: Edge ANPR Toll Booth System on RDK X5

Production-ready, hardware-accelerated Automatic Number Plate Recognition (ANPR) and automated toll billing system deployed on the **D-Robotics RDK X5 (Horizon Sunrise 5 / 10 TOPS BPU)**.

---

## ⚡ System Highlights

* **Hardware Acceleration**: YOLOv8n license plate detection executes on the **10 TOPS BPU** in **~5ms per frame** using `hobot_dnn.pyeasy_dnn`.
* **Zero-Lag Architecture**: Plate detection runs on BPU, while OCR is processed asynchronously in a **background worker thread** — video streaming runs at full **10–15+ FPS** without stalling.
* **Edge OCR Engine**: Powered by **RapidOCR (ONNX Runtime backend)**, eliminating native library conflicts with Horizon SDK while achieving high accuracy on ARM64.
* **Smart Region Tracking**: Tracks vehicles across frames via IoU to select the single sharpest, highest-confidence crop per vehicle before running OCR.
* **Indian HSRP Compliance**: Validates Indian state codes (including BH series) and automatically corrects common OCR character confusions.
* **Modern Web Dashboard**: Real-time responsive web dashboard built with Flask, MJPEG streaming, and FontAwesome icons — accessible from laptops, tablets, or smartphones on the local network.
* **Pre-compiled BPU Model**: Includes the ready-to-run compiled model: `models/yolov8n_plate_bayese_640x640_nv12.bin`.

---

## 🏗️ Architecture

```
GS130W MIPI Camera / USB Cam / Video File
                    │
                    ▼
     YOLOv8n Detector (BPU Core: ~5ms)
                    │
                    ▼
          Smart Region Tracker (IoU)
                    │ (Submits best crop)
                    ▼
      Background OCR Worker (RapidOCR)
                    │
                    ▼
         Indian Plate Validator (HSRP)
                    │
                    ▼
       SQLite Database & Deduplication
                    │
                    ▼
    Flask Web Dashboard (http://<rdk-ip>:5000)
```

---

## 📂 Directory Structure

```text
step_3_anpr_rdkx5/
├── models/
│   └── yolov8n_plate_bayese_640x640_nv12.bin   # Pre-compiled 10 TOPS BPU INT8 model
├── src/
│   ├── camera.py                               # GS130W MIPI CSI & OpenCV video stream capture
│   ├── database.py                             # SQLite toll database & deduplication manager
│   ├── detector.py                             # YOLOv8 BPU runtime detector (pyeasy_dnn)
│   ├── plate_validator.py                      # Indian plate validation & auto-fix rules
│   ├── recognizer.py                           # RapidOCR edge text recognition engine
│   └── utils.py                                # BGR/NV12 transforms, drawing, similarity
├── web/
│   ├── app.py                                  # Flask web server & MJPEG streaming endpoint
│   ├── static/                                 # Modern CSS stylesheet & JavaScript
│   └── templates/
│       └── dashboard.html                      # Mobile-responsive dark-theme UI
├── data/
│   ├── anpr.db                                 # SQLite database (auto-created on first run)
│   └── plate_crops/                            # Saved plate crop images
├── main.py                                     # Master pipeline orchestration script
├── requirements.txt                            # Python dependencies (RapidOCR, Flask, etc.)
├── setup.sh                                    # Automated environment installation script
└── README.md
```

---

## 🔌 Power & Hardware Setup

> [!IMPORTANT]
> The RDK X5 requires a dedicated **5V / 3A (15W)** or **5V / 5A (25W)** power supply via USB Type-C (e.g. Raspberry Pi 4 power adapter or ERD 5V 3A supply). Generic phone chargers (5V 2A / 10W) will experience voltage sag when the camera and BPU activate, causing camera initialization failure.

### Camera Connection (GS130W MIPI):
* **Cable Directionality**: Ensure the cable end marked `CAM` is connected to the camera board, and the end marked `RDK` is plugged into the RDK X5 board.
* **Port**: Connect to **CAM0** (or **CAM1** if using `--source mipi1`).
* **Cold Boot**: Always connect the MIPI ribbon cable while the board is completely powered off.

---

## 🚀 Quick Start on RDK X5

### 1. Clone or Pull on RDK X5:
```bash
git clone -b step-3-rdk-deployment https://github.com/ShahbazCoder1/ANPR-System-RDK-X5.git
cd ANPR-System-RDK-X5/step_3_anpr_rdkx5
```

### 2. Run Setup:
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Launch the System:

#### Mode A: Prerecorded Video Demonstration
```bash
python3 main.py --source test_video.mp4 --conf 0.40 --toll 100
```

#### Mode B: USB Webcam
```bash
python3 main.py --source 0 --conf 0.40 --toll 100
```

#### Mode C: GS130W MIPI Camera
```bash
# For CAM0 port:
python3 main.py --source mipi --conf 0.40 --toll 100

# For CAM1 port:
python3 main.py --source mipi1 --conf 0.40 --toll 100
```

---

## 🌐 Web Dashboard

Once running, access the dashboard from any browser on the same Wi-Fi or LAN:
```
http://<RDK_X5_IP>:5000
```

* **Live Feed**: 60 FPS MJPEG stream with detection bounding boxes & plate overlays.
* **Real-Time KPIs**: Total vehicles, daily toll revenue collected, pipeline FPS, and system uptime.
* **Toll Records**: Auto-updating table with plate crops, recognized numbers, confidence scores, and timestamps.
