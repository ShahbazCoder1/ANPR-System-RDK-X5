# RDK X5 Edge AI Automatic Number Plate Recognition (ANPR) System

[![Hardware](https://img.shields.io/badge/Hardware-D--Robotics%20RDK%20X5%20(10%20TOPS)-blue.svg)](https://developer.d-robotics.cc/)
[![Model](https://img.shields.io/badge/Model-YOLOv8n%20%7C%20RapidOCR-green.svg)](https://github.com/ultralytics/ultralytics)
[![Quantization](https://img.shields.io/badge/Quantization-INT8%20(Horizon%20PTQ)-orange.svg)](https://developer.d-robotics.cc/)
[![Interface](https://img.shields.io/badge/UI-Flask%20Live%20Dashboard-purple.svg)](https://flask.palletsprojects.com/)

An end-to-end, hardware-accelerated **Edge AI Automatic Number Plate Recognition (ANPR) and Automated Toll Booth System** developed for the **D-Robotics RDK X5** single-board computer (equipped with a **10 TOPS Horizon Sunrise 5 BPU**).

---

## 📌 Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             END-TO-END WORKFLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

  [ STEP 1: TRAINING & TESTING ]
   Dataset (Roboflow / YOLO)
              │
              ▼
   YOLOv8n Training (RTX 3050 4GB GPU) ────► Evaluation & Local Test Pipeline
              │
              ▼ (best.pt)
  [ STEP 2: MODEL CONVERSION ]
   Export to ONNX (Opset 11)
              │
              ▼
   Horizon OpenExplorer Docker ───────────► Calibration (PTQ INT8)
              │
              ▼ (yolov8n_plate_bayese_640x640_nv12.bin)
  [ STEP 3: EDGE DEPLOYMENT ]
   D-Robotics RDK X5 (10 TOPS BPU)
              │
              ├──► BPU YOLOv8n Inference (~5ms)
              ├──► Smart Vehicle Region Tracker (IoU)
              ├──► Background Multi-Engine Edge OCR (RapidOCR)
              ├──► Indian HSRP Format Validation & Auto-Correction
              ├──► Local SQLite Deduplication & Database
              └──► Live Mobile-Responsive Web Dashboard (Port 5000)
```

---

## 🗂️ Repository Structure

This repository is structured into three self-contained, step-by-step modular folders:

```text
ANPR-System-RDK-X5/
│
├── step_1_model_training/                     # STEP 1: Model Training & Validation
│   ├── README.md                              # Training instructions & GPU tuning guide
│   ├── train.py                               # YOLOv8n training script (RTX 3050 optimizations)
│   ├── test_model.py                          # Evaluation tool (dataset metrics, image & video tests)
│   ├── anpr_pipeline.py                       # Local workstation ANPR test pipeline
│   ├── setup_env.bat                          # PyTorch + CUDA environment setup
│   ├── install_ocr.bat                        # OCR setup script
│   └── yolov8n.pt                             # Base YOLOv8n pre-trained weights
│
├── step_2_model_conversion/                   # STEP 2: ONNX & BPU .bin Model Export
│   ├── README.md                              # Docker OpenExplorer PTQ quantization guide
│   ├── export_model.py                        # Exports best.pt -> simplified ONNX (opset 11)
│   ├── configs/
│   │   └── yolov8n_plate_config.yaml          # Horizon compiler configuration for bayes-e core
│   ├── scripts/
│   │   ├── convert_to_bin.sh                  # Automation script running hb_mapper in Docker
│   │   └── prepare_calibration.py             # Generates NV12 calibration samples from dataset
│   └── onnx_models/
│       └── best.onnx                          # Exported ONNX model
│
└── step_3_anpr_rdkx5/                         # STEP 3: Edge ANPR Deployment on RDK X5
    ├── README.md                              # RDK X5 hardware guide, wiring & CLI documentation
    ├── main.py                                # Master pipeline with asynchronous OCR worker
    ├── setup.sh                               # RDK X5 board environment setup script
    ├── requirements.txt                       # Board dependencies (RapidOCR, Flask, hobot_dnn)
    ├── models/
    │   └── yolov8n_plate_bayese_640x640_nv12.bin  # Compiled BPU binary model (~5ms inference)
    ├── src/
    │   ├── camera.py                          # GS130W MIPI CSI & OpenCV video stream capture
    │   ├── detector.py                        # BPU YOLO runtime inference engine (pyeasy_dnn)
    │   ├── recognizer.py                      # RapidOCR edge text recognition engine
    │   ├── database.py                        # SQLite toll database & deduplication manager
    │   ├── plate_validator.py                 # Indian HSRP regex & state code auto-corrector
    │   └── utils.py                           # NV12 conversion, letterboxing & HUD visualization
    └── web/
        ├── app.py                             # Flask web server & MJPEG streaming endpoint
        ├── templates/
        │   └── dashboard.html                 # Mobile-responsive web dashboard with FontAwesome
        └── static/
            ├── css/style.css                  # Polished dark theme responsive stylesheet
            └── js/dashboard.js                # Live polling & UI update scripts
```

---

## ⚡ Performance Benchmarks

| Metric | Workstation (RTX 3050) | Edge Device (RDK X5 BPU) |
| :--- | :--- | :--- |
| **YOLOv8 Detection Latency** | ~6.5 ms (CUDA FP16) | **~5.1 ms (BPU INT8)** |
| **Plate Detection mAP@50** | 98.4% | 98.1% (Lossless PTQ) |
| **OCR Recognition Backend** | PaddleOCR (GPU) | RapidOCR (ARM64 ONNX) |
| **Video Stream Pipeline FPS** | 25-30 FPS | **10-15+ FPS (Non-blocking worker)** |
| **Power Consumption** | ~120 Watts | **~8-10 Watts** |

---

## 🚀 Quick Navigation

* **To train or validate the model:** See **[`step_1_model_training/README.md`](step_1_model_training/README.md)**
* **To convert models using Docker:** See **[`step_2_model_conversion/README.md`](step_2_model_conversion/README.md)**
* **To run the live system on RDK X5:** See **[`step_3_anpr_rdkx5/README.md`](step_3_anpr_rdkx5/README.md)**

---

## 📄 License
This project is licensed under the Apache 2.0 License.
