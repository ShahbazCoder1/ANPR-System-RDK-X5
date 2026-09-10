# RDK X5 Edge AI Automatic Number Plate Recognition (ANPR) System

[![Hardware](https://img.shields.io/badge/Hardware-D--Robotics%20RDK%20X5%20(10%20TOPS)-blue.svg)](https://developer.d-robotics.cc/)
[![Model](https://img.shields.io/badge/Model-YOLOv8n%20%7C%20RapidOCR-green.svg)](https://github.com/ultralytics/ultralytics)
[![Quantization](https://img.shields.io/badge/Quantization-INT8%20(Horizon%20PTQ)-orange.svg)](https://developer.d-robotics.cc/)
[![Interface](https://img.shields.io/badge/UI-Flask%20Live%20Dashboard-purple.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

A production-ready, hardware-accelerated **Edge AI Automatic Number Plate Recognition (ANPR) and Automated Toll Booth System** designed for instant deployment on the **D-Robotics RDK X5** single-board computer (equipped with a **10 TOPS Horizon Sunrise 5 BPU**).

---

> [!NOTE]
> **Looking for Model Training & Conversion Code?**
> This `main` branch contains the clean, ready-to-run edge deployment system with the pre-compiled 10 TOPS BPU model included.
>
> If you want to train your own custom license plate model from scratch or convert `.pt` weights into `.bin` via Horizon OpenExplorer Docker, please switch to the **[`full-pipeline`](https://github.com/ShahbazCoder1/ANPR-System-RDK-X5/tree/full-pipeline)** branch:
> * **`step_1_model_training/`**: PyTorch YOLOv8n training & evaluation scripts (RTX 3050 optimizations).
> * **`step_2_model_conversion/`**: ONNX export, calibration, and Docker compilation configs.
> * **`step_3_anpr_rdkx5/`**: Standalone modular edge deployment.

---

## ⚡ Key Features

* **10 TOPS BPU Acceleration**: YOLOv8n license plate detection runs on-chip in **~5ms per frame** using `hobot_dnn.pyeasy_dnn`.
* **Zero-Latency Video Pipeline**: BPU detection and OCR recognition are decoupled using an asynchronous **background worker thread** — video streaming runs at **10–15+ FPS** without frame drops or pauses.
* **RapidOCR On-Edge Engine**: High-accuracy text recognition optimized for ARM64 with an ONNX Runtime backend (no conflicting shared libraries).
* **Smart Region Tracking**: Uses IoU tracking to pick the sharpest, highest-confidence plate crop per passing vehicle, preventing redundant OCR operations.
* **Indian HSRP Compliance**: Validates standard Indian license plates (including BH series) and auto-corrects common OCR character confusions (e.g. `O` $\leftrightarrow$ `0`, `I` $\leftrightarrow$ `1`, `B` $\leftrightarrow$ `8`).
* **Mobile-Responsive Web Dashboard**: Real-time Flask dashboard with live MJPEG streaming, real-time vehicle logs, toll revenue statistics, and FontAwesome icons.
* **Pre-Compiled BPU Model Included**: The compiled INT8 binary model (`models/yolov8n_plate_bayese_640x640_nv12.bin`) is pre-packaged and ready to run immediately upon cloning!

---

## 🏗️ System Pipeline

```
GS130W MIPI Camera / USB Cam / Video File
                    │
                    ▼
     YOLOv8n Detector (BPU Core: ~5ms)
                    │
                    ▼
          Smart Region Tracker (IoU)
                    │ (Selects best crop)
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

## 📂 Repository Structure

```text
ANPR-System-RDK-X5/ (branch: main)
├── models/
│   └── yolov8n_plate_bayese_640x640_nv12.bin   # Pre-compiled 10 TOPS BPU INT8 model binary
├── src/
│   ├── camera.py                               # GS130W MIPI CSI & OpenCV video stream capture
│   ├── detector.py                             # BPU YOLO runtime inference engine (pyeasy_dnn)
│   ├── recognizer.py                           # RapidOCR edge text recognition engine
│   ├── database.py                             # SQLite toll database & deduplication manager
│   ├── plate_validator.py                      # Indian HSRP regex & state code auto-corrector
│   └── utils.py                                # NV12 conversion, letterboxing & HUD visualization
├── web/
│   ├── app.py                                  # Flask web server & MJPEG streaming endpoint
│   ├── static/                                 # Modern dark theme CSS stylesheet & JS polling
│   └── templates/
│       └── dashboard.html                      # Mobile-responsive web dashboard with FontAwesome
├── data/
│   ├── anpr.db                                 # SQLite database (auto-generated)
│   └── plate_crops/                            # Saved plate crop images
├── main.py                                     # Master pipeline orchestration script
├── requirements.txt                            # Python dependencies (RapidOCR, Flask, hobot_dnn)
├── setup.sh                                    # 1-command environment installer
└── README.md
```

---

## 🔌 Hardware & Power Requirements

> [!IMPORTANT]
> The RDK X5 requires a dedicated **5V / 3A (15W)** or **5V / 5A (25W)** power supply via USB Type-C (e.g. Raspberry Pi 4 power brick or ERD 5V 3A supply). 
> Generic phone chargers (5V 2A / 10W) will suffer voltage sag when the camera and BPU activate, causing camera initialization failure (`0x00`).

### Camera Setup (GS130W MIPI):
* **Cable Direction**: The end of the ribbon cable marked `CAM` must plug into the camera module; the end marked `RDK` plugs into the board.
* **Port**: Connect to **CAM0** by default (or **CAM1** if using `--source mipi1`).
* **Power Cycle**: Always connect the MIPI ribbon cable while the board is completely powered off.

---

## 🚀 1-Command Quickstart on RDK X5

### 1. Clone the Repository:
```bash
git clone https://github.com/ShahbazCoder1/ANPR-System-RDK-X5.git
cd ANPR-System-RDK-X5
```

### 2. Run Environment Setup:
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Launch the System:

#### Mode A: Prerecorded Video Demonstration
```bash
python3 main.py --source test_video.mp4 --conf 0.40 --toll 100
```

#### Mode B: Standard USB Webcam
```bash
python3 main.py --source 0 --conf 0.40 --toll 100
```

#### Mode C: GS130W MIPI CSI Camera
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

* **Live Video Feed**: Low-latency MJPEG stream with detection bounding boxes & plate overlays.
* **Real-Time KPIs**: Total vehicles passed, daily toll revenue collected, pipeline FPS, and system uptime.
* **Toll Records**: Auto-updating table displaying plate crops, recognized license numbers, confidence scores, and timestamps.
* **Responsive Layout**: Designed for seamless viewing across smartphones, tablets, and desktop monitors.

---

## 📄 License
This project is licensed under the Apache 2.0 License.
