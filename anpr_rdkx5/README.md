# RDK X5 ANPR Toll Booth System (BPU-Accelerated)

Complete edge AI deployment of an Automatic Number Plate Recognition (ANPR) Toll Booth System running on the **D-Robotics RDK X5 (Sunrise 5 / 10 TOPS BPU)**.

---

## System Architecture

```
GS130W MIPI Camera / Video File
             │
             ▼
    YOLOv8n Plate Detector (BPU: ~5ms)
             │
             ▼
   PaddleOCR Text Recognizer (Multi-crop: Upscale + CLAHE + Sharpen)
             │
             ▼
 Indian Plate Validator (State Code Auto-Fix + RegEx)
             │
             ▼
  SQLite Database (anpr.db - Deduplication & Toll Fee Recording)
             │
             ▼
 Flask Web Dashboard (Live Video Feed + Recent Detections + Stats)
             │
             ▼
 Browser (Laptop / Phone on same WiFi @ http://<rdk-ip>:5000)
```

---

## Directory Structure

```text
anpr_rdkx5/
├── models/
│   └── yolov8n_plate_bayese_640x640_nv12.bin   # BPU compiled model (copied here)
├── src/
│   ├── camera.py                               # GS130W MIPI CSI & video file capture
│   ├── database.py                             # SQLite toll database manager
│   ├── detector.py                             # YOLOv8n BPU detector (hbm_runtime)
│   ├── plate_validator.py                      # Indian plate validation & auto-fix
│   ├── recognizer.py                           # PaddleOCR text recognizer
│   └── utils.py                                # BGR/NV12 transforms, drawing, similarity
├── web/
│   ├── app.py                                  # Flask web server & MJPEG stream
│   ├── static/                                 # CSS and JavaScript
│   └── templates/
│       └── dashboard.html                      # Real-time Toll Booth Web UI
├── data/
│   ├── anpr.db                                 # SQLite database (auto-generated)
│   └── plate_crops/                            # Saved plate crop images
├── main.py                                     # Master pipeline orchestration script
├── requirements.txt                            # Board Python dependencies
├── setup.sh                                    # 1-command environment installer
└── README.md
```

---

## Quick Start on RDK X5

### 1. Copy the `anpr_rdkx5/` Folder to RDK X5
From your laptop, copy the folder to your RDK X5 home directory:
```bash
scp -r anpr_rdkx5 sunrise@<rdk-x5-ip>:~/
```

### 2. Run Setup on the Board
SSH into your RDK X5:
```bash
ssh sunrise@<rdk-x5-ip>
cd ~/anpr_rdkx5
chmod +x setup.sh
./setup.sh
```

### 3. Place your Compiled BPU Model
Ensure your compiled model from the Ubuntu VM conversion is placed at:
`anpr_rdkx5/models/yolov8n_plate_bayese_640x640_nv12.bin`

### 4. Run the Pipeline

#### Phase 1: Test with a Video File
```bash
python3 main.py --source test_video.mp4 --conf 0.60 --toll 100
```

#### Phase 2: Run with GS130W MIPI Stereo Camera
```bash
python3 main.py --source mipi --conf 0.60 --toll 100
```

---

## Accessing the Live Web Dashboard

From any laptop, tablet, or phone connected to the same Wi-Fi / LAN network:
Open your browser and navigate to:
```
http://<rdk-x5-ip>:5000
```

### Dashboard Features:
1. **Live Camera Feed**: Real-time MJPEG stream with YOLO plate boxes and OCR tags.
2. **Real-time Toll Stats**: Total vehicles detected today, total toll collected (₹), current BPU FPS, and system uptime.
3. **Recent Records Table**: Auto-updating table showing plate crop thumbnails, Indian license plate badges, detection timestamps, toll fees, and confidence scores.
