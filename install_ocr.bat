@echo off
echo ===================================================
echo Installing PaddleOCR for ANPR Pipeline
echo ===================================================

echo Installing PaddlePaddle and PaddleOCR...
pip install paddlepaddle paddleocr opencv-python pyyaml pandas

echo.
echo ===================================================
echo Verifying PaddleOCR Installation...
echo ===================================================
python -c "from paddleocr import PaddleOCR; print('PaddleOCR installed successfully!')"

echo.
echo Setup Complete!
pause
