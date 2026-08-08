@echo off
echo ===================================================
echo Setting up Python Environment for YOLOv8 Training
echo ===================================================

echo Installing PyTorch with CUDA 12.4 support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo Installing Ultralytics, OpenCV, and OpenCV-Python...
pip install ultralytics opencv-python pyyaml onnx onnxscript

echo.
echo ===================================================
echo Verifying Installation...
echo ===================================================
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU Detected')"

echo.
echo Setup Complete!
pause
