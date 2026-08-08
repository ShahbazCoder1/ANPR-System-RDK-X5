@echo off
echo ===================================================
echo Installing EasyOCR for ANPR Pipeline
echo ===================================================

echo Installing EasyOCR and dependencies...
pip install easyocr opencv-python pyyaml pandas

echo.
echo ===================================================
echo Verifying EasyOCR Installation...
echo ===================================================
python -c "import easyocr; print('EasyOCR Version:', easyocr.__version__); print('EasyOCR installed successfully!')"

echo.
echo Setup Complete!
pause
