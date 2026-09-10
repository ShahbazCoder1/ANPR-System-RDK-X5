#!/bin/bash
set -e

echo "========================================================"
echo "    Horizon OpenExplorer: YOLOv8n ONNX -> BIN Converter"
echo "    Target: RDK X5 (Horizon Sunrise 5 / bayes-e BPU)   "
echo "========================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$BASE_DIR/output_bins"

echo ""
echo "[Step 1/2] Checking model operator compatibility for bayes-e..."
hb_mapper checker \
  --model-type onnx \
  --march bayes-e \
  --model "$BASE_DIR/onnx_models/best.onnx"

echo ""
echo "[Step 2/2] Running Post-Training Quantization (PTQ) & compilation..."
hb_mapper makertbin \
  --model-type onnx \
  --config "$BASE_DIR/configs/yolov8n_plate_config.yaml"

echo ""
echo "========================================================"
echo " [SUCCESS] Model compilation completed!"
echo " Compiled .bin file is in: $BASE_DIR/output_bins/"
echo " Copy the .bin file into your RDK X5 anpr_rdkx5/models/ directory."
echo "========================================================"
