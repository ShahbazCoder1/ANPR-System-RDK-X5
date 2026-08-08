@echo off
echo ===================================================
echo Installing PaddleOCR for ANPR Pipeline
echo ===================================================

echo Installing PaddlePaddle and PaddleOCR (v2.9.1 for Windows compatibility)...
pip install paddlepaddle "paddleocr==2.9.1" opencv-python pyyaml pandas

echo.
echo ===================================================
echo Verifying PaddleOCR Installation...
echo ===================================================
python -c "from paddleocr import PaddleOCR; print('PaddleOCR installed successfully!')"

echo.
echo Setup Complete!
pause
