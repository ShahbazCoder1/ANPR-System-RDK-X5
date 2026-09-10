# Step 2: YOLOv8n ONNX & Horizon BPU .bin Model Export

This directory contains the tools and configuration needed to convert your trained PyTorch weights (`best.pt`) into an **INT8 quantized BPU binary (`yolov8n_plate_bayese_640x640_nv12.bin`)** optimized for the **10 TOPS BPU** on the **D-Robotics RDK X5 (Horizon Sunrise 5 / bayes-e)**.

---

## 📂 Directory Contents

| File / Folder | Description |
| :--- | :--- |
| `export_model.py` | Exports PyTorch `best.pt` to simplified ONNX format (`opset=11`, `640x640`) |
| `configs/yolov8n_plate_config.yaml` | Horizon `hb_mapper` compilation & post-training quantization (PTQ) config |
| `scripts/prepare_calibration.py` | Samples 100 images from dataset and converts them to raw NV12 calibration binaries |
| `scripts/convert_to_bin.sh` | Shell script running `hb_mapper checker` and `hb_mapper makertbin` inside Docker |
| `onnx_models/best.onnx` | Stored ONNX model ready for BPU compilation |

---

## 🛠️ Step-by-Step Conversion Workflow

### Stage 1: Export PyTorch to ONNX (on your Local PC)

Run `export_model.py` to convert `best.pt` to an ONNX model formatted specifically for the Horizon BPU compiler:
```bash
# Automatically finds best.pt from Step 1
python export_model.py

# Or specify custom weights path:
python export_model.py path/to/best.pt
```

**Key Export Parameters:**
* Format: `ONNX`
* Resolution: `640x640`
* Opset: `11` (Strictly required for Horizon OpenExplorer compatibility)
* Simplification: `True` (Removes redundant graph nodes via onnxsim)
* Output location: `onnx_models/best.onnx`

---

### Stage 2: Prepare Calibration Dataset (on your Local PC)

Horizon BPU uses Post-Training Quantization (PTQ) to quantize FP32 model weights down to INT8 with zero accuracy degradation. To calibrate the quantization thresholds:
```bash
python scripts/prepare_calibration.py
```
*This selects 100 representative images from your dataset, normalizes them, and outputs raw uint8 binary images into `calibration_data/`.*

---

### Stage 3: Compile to BPU `.bin` using Docker (Ubuntu VM / Linux)

The Horizon OpenExplorer toolchain runs inside a dedicated Docker container.

#### 1. Copy `step_2_model_conversion` to your Ubuntu machine:
```bash
scp -r step_2_model_conversion user@your-ubuntu-vm:~/
```

#### 2. Install Docker (if not installed):
```bash
sudo apt update && sudo apt install -y docker.io
sudo usermod -aG docker $USER
# Log out and log back in, or run: newgrp docker
```

#### 3. Download & Load the Official Horizon OpenExplorer Docker Image:
```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker images
```

#### 4. Launch the Compilation Container:
```bash
cd ~/step_2_model_conversion

docker run -it --rm \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

#### 5. Execute Compilation Inside the Container:
```bash
chmod +x scripts/convert_to_bin.sh
bash scripts/convert_to_bin.sh
```

**What happens inside `convert_to_bin.sh`:**
1. `hb_mapper checker`: Validates that every YOLOv8 layer is supported by the `bayes-e` hardware core.
2. `hb_mapper makertbin`: Quantizes weights to INT8 using calibration samples, optimizes tensor layouts for NV12 input, and packages the model into a `.bin` file.

---

### Stage 4: Retrieve Compiled Model

The compiled BPU binary model will be saved at:
```
output_bins/yolov8n_plate_bayese_640x640_nv12.bin
```

Copy this `.bin` model file into your deployment directory:
```bash
cp output_bins/yolov8n_plate_bayese_640x640_nv12.bin ../step_3_anpr_rdkx5/models/
```

Proceed to **[Step 3: Edge ANPR Deployment](../step_3_anpr_rdkx5/README.md)**!
