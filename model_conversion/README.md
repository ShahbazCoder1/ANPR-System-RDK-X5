# ONNX to BIN Model Conversion Guide (for RDK X5)

This folder contains all files and configurations needed to convert your trained YOLOv8n license plate detector (`best.onnx`) into the BPU-accelerated INT8 `.bin` format (`yolov8n_plate_bayese_640x640_nv12.bin`) for the **RDK X5 (Sunrise 5 / bayes-e)** board.

---

## Prerequisites (on your Ubuntu VM)

1. **Docker Installed**:
   ```bash
   sudo apt update
   sudo apt install -y docker.io
   sudo usermod -aG docker $USER
   # Log out and log back in (or run 'newgrp docker')
   ```

2. **Hardware Requirements for VM**:
   - Minimum 8 GB RAM (12 GB recommended)
   - At least 15 GB free disk space

---

## Step-by-Step Conversion Guide

### Step 1: Copy `model_conversion/` to your Ubuntu VM
Copy this entire `model_conversion/` folder from Windows into your Ubuntu VM home directory:
```bash
# Example via scp or shared folder:
cd ~/model_conversion
ls -la
```

### Step 2: Download the Official OpenExplorer Docker Image
Download the Horizon OpenExplorer 1.2.8 CPU image for RDK X5:
```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker images
```

### Step 3: Run the Conversion inside Docker Container
Launch the container, mounting the `model_conversion` directory:
```bash
docker run -it --rm \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

Inside the Docker shell:
```bash
# Run the complete conversion script:
chmod +x scripts/convert_to_bin.sh
bash scripts/convert_to_bin.sh
```

### Step 4: Retrieve the Compiled Model
The compiled file will be located at:
`output_bins/yolov8n_plate_bayese_640x640_nv12.bin`

Copy this `.bin` file into your `anpr_rdkx5/models/` folder for deployment on the RDK X5 board!
