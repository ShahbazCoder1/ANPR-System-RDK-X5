import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np

def prepare_calibration_data(num_samples=100, target_size=(640, 640)):
    """Sample representative images from training set for PTQ calibration."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    src_dir = base_dir / "License Plate Detection" / "train" / "images"
    out_dir = base_dir / "model_conversion" / "calibration_data"
    
    if not src_dir.exists():
        print(f"[ERROR] Source image directory not found: {src_dir}")
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [f for f in src_dir.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not all_images:
        print(f"[ERROR] No images found in {src_dir}")
        return
        
    print(f"Total training images found: {len(all_images)}")
    selected = random.sample(all_images, min(num_samples, len(all_images)))
    print(f"Sampling {len(selected)} images for calibration...")
    
    # Clean any old files
    for old_f in out_dir.glob("*"):
        old_f.unlink()

    for i, img_path in enumerate(selected):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to 640x640 for YOLO model calibration
        resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        # Transpose from HWC (640, 640, 3) to NCHW (3, 640, 640)
        img_nchw = np.transpose(resized, (2, 0, 1))
        
        # Save as raw uncompressed uint8 binary file (1,228,800 bytes)
        dst_path = out_dir / f"calib_{i:04d}.bin"
        img_nchw.astype(np.uint8).tofile(str(dst_path))
        
    print(f"[SUCCESS] Calibration dataset generated in: {out_dir}")
    print(f"Total calibration binary samples: {len(list(out_dir.glob('*.bin')))}")

if __name__ == "__main__":
    prepare_calibration_data()
