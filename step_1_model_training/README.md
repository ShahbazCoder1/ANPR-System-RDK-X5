# Step 1: License Plate Detection Model Training & Testing

This directory contains everything required to train, validate, and test a custom **YOLOv8n** license plate detection model on your local PC / workstation (optimized for budget GPUs such as the NVIDIA RTX 3050 4GB VRAM).

---

## 📂 Directory Contents

| File / Folder | Description |
| :--- | :--- |
| `train.py` | Training script with memory optimizations (Batch=8, AMP=True, Workers=2) |
| `test_model.py` | Model evaluation tool supporting dataset metrics (mAP50/mAP50-95), single image inference, and video inference |
| `anpr_pipeline.py` | Workstation ANPR pipeline prototype (YOLO detection + PaddleOCR recognition + Indian plate format validator) |
| `setup_env.bat` | Windows batch setup script for PyTorch (CUDA 12.4), Ultralytics, and OpenCV |
| `install_ocr.bat` | Windows batch setup script for PaddleOCR / PaddlePaddle |
| `yolov8n.pt` | Official Ultralytics base weights for transfer learning |

---

## ⚙️ Prerequisites & Setup

### 1. Hardware Requirements
* **GPU**: NVIDIA GPU with 4GB+ VRAM (e.g. RTX 3050, GTX 1650, RTX 2060, etc.)
* **RAM**: 8 GB minimum (16 GB recommended)
* **Storage**: 10 GB free disk space

### 2. Environment Setup (Windows)
Run the automated batch script to set up PyTorch with CUDA 12.4 and Ultralytics:
```cmd
setup_env.bat
```

Or manually install via pip:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics opencv-python pyyaml onnx onnxscript
```

To install OCR packages for the local testing pipeline:
```cmd
install_ocr.bat
```

---

## 📊 Dataset Structure

Place your YOLOv8-formatted dataset in a folder named `License Plate Detection` (either inside this directory or at the project root):

```
License Plate Detection/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**`data.yaml` Example:**
```yaml
path: ../License Plate Detection
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['license-plate']
```

---

## 🚀 1. Train the YOLOv8n Plate Detector

Run the training script:
```bash
python train.py
```

### Key Training Optimizations (for 4GB VRAM):
* **Model**: YOLOv8 Nano (`yolov8n.pt`)
* **Input Resolution**: `640x640`
* **Batch Size**: `8` (prevents CUDA Out-Of-Memory errors on 4GB VRAM)
* **AMP (Automatic Mixed Precision)**: `True` (FP16 math doubles speed and reduces VRAM usage)
* **Optimizer**: `AdamW` (learning rate `lr0=0.001`)
* **Epochs**: `100` with early stopping `patience=20`

Training artifacts and the best weights will be saved to:
```
runs/detect_plate/weights/best.pt
```

---

## 🧪 2. Test & Evaluate the Trained Model

Use `test_model.py` to evaluate performance across three modes:

### Mode A: Dataset Validation Metrics (mAP50, Precision, Recall)
```bash
python test_model.py --mode dataset
```
*Evaluates against the held-out test split and outputs mAP50, mAP50-95, Precision, and Recall.*

### Mode B: Single Image Inference
```bash
python test_model.py --mode image --source path/to/car.jpg --conf 0.50
```

### Mode C: Video Inference & Annotation
```bash
python test_model.py --mode video --source path/to/traffic_video.mp4 --conf 0.50
```
*Processes the video frame-by-frame and exports the annotated video to `runs/video_output/`.*

---

## 🚗 3. Test Full Workstation ANPR Pipeline

Test end-to-end license plate detection, OCR extraction, and Indian plate format validation on your PC:
```bash
python anpr_pipeline.py --source path/to/traffic_video.mp4 --conf 0.50 --save-crops
```

---

## ➡️ Next Step

Once you are satisfied with your trained weights in `runs/detect_plate/weights/best.pt`, proceed to **[Step 2: Model Conversion](../step_2_model_conversion/README.md)** to export to ONNX and compile the BPU `.bin` model for RDK X5!
