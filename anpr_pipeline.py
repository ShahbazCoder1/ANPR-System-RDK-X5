import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

# Suppress PyTorch / Ultralytics verbose logs where possible
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Valid 2-letter State & UT Codes in India (including BH for Bharat Series)
VALID_INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB", "BH"
}

# Indian License Plate RegEx: State(2) + District(1-2) + Series(1-3) + Number(4)
INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

def clean_and_validate_indian_plate(raw_text: str) -> str | None:
    """Clean raw OCR text, auto-correct character confusions, and strictly validate Indian State Code + Format."""
    if not raw_text:
        return None
    
    # Uppercase and strip whitespace / non-alphanumeric characters
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())

    # Length check: Indian plates are between 8 and 11 characters
    if not (8 <= len(cleaned) <= 11):
        return None

    # Common character confusion mapping based on positional rules
    fixed = list(cleaned)

    # 1. First 2 characters MUST be a valid State Code (Letters only)
    for i in range(2):
        if fixed[i] == '0': fixed[i] = 'O'
        elif fixed[i] == '1': fixed[i] = 'I'
        elif fixed[i] == '8': fixed[i] = 'B'
        elif fixed[i] == '5': fixed[i] = 'S'
        elif fixed[i] == '4': fixed[i] = 'A'

    state_code = "".join(fixed[:2])
    if state_code not in VALID_INDIAN_STATES:
        # Try minor fixes for state code (e.g., M0 -> MH, K4 -> KA, D1 -> DL, M8 -> MH)
        state_fixes = {"M0": "MH", "M8": "MH", "MJ": "MH", "MB": "MH", "ME": "MH",
                       "D1": "DL", "D0": "DL", "K4": "KA", "K8": "KA", "KB": "KA",
                       "W8": "WB", "Y0": "UP", "YE": "UP"}
        if state_code in state_fixes:
            fixed[0], fixed[1] = state_fixes[state_code][0], state_fixes[state_code][1]
        else:
            return None  # Invalid state code -> Reject false detection

    # 2. District code (Next 1-2 chars MUST be digits)
    # Check digits for index 2 and 3
    if len(fixed) >= 4:
        for i in (2, 3):
            if i < len(fixed) - 4:  # Do not touch the last 4 digits
                if fixed[i].isdigit() is False:
                    if fixed[i] == 'O': fixed[i] = '0'
                    elif fixed[i] == 'I': fixed[i] = '1'
                    elif fixed[i] == 'Z': fixed[i] = '2'
                    elif fixed[i] == 'S': fixed[i] = '5'
                    elif fixed[i] == 'B': fixed[i] = '8'

    # 3. Last 4 characters MUST be digits
    for i in range(len(fixed) - 4, len(fixed)):
        if fixed[i].isdigit() is False:
            if fixed[i] == 'O': fixed[i] = '0'
            elif fixed[i] == 'I': fixed[i] = '1'
            elif fixed[i] == 'Z': fixed[i] = '2'
            elif fixed[i] == 'S': fixed[i] = '5'
            elif fixed[i] == 'B': fixed[i] = '8'

    fixed_str = "".join(fixed)
    if fixed_str[:2] in VALID_INDIAN_STATES and INDIAN_PLATE_REGEX.match(fixed_str):
        return fixed_str

    return None

def preprocess_plate_crop(crop_img: np.ndarray) -> np.ndarray:
    """Enhanced preprocessing for OCR: Upscale + Grayscale + CLAHE Contrast Enhancement."""
    if crop_img is None or crop_img.size == 0:
        return crop_img

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # 2. Resize / Upscale to standard width (350px) for crisp character recognition
    h, w = gray.shape[:2]
    if w > 0:
        target_w = 350
        target_h = int(h * (target_w / float(w)))
        gray = cv2.resize(gray, (target_w, max(target_h, 70)), interpolation=cv2.INTER_CUBIC)

    # 3. CLAHE Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # 4. Light Bilateral Filter to smooth noise while preserving sharp character edges
    denoised = cv2.bilateralFilter(contrast_enhanced, 5, 50, 50)

    return denoised

def run_anpr_pipeline():
    parser = argparse.ArgumentParser(description="End-to-End ANPR Pipeline (YOLOv8 + EasyOCR + Strict RegEx)")
    parser.add_argument("--source", type=str, required=True, help="Path to video file or camera index (e.g. 0)")
    parser.add_argument("--weights", type=str, default="", help="Custom YOLO weights path (defaults to runs/detect_plate/weights/best.pt)")
    parser.add_argument("--conf", type=float, default=0.60, help="YOLO plate detection confidence threshold (default: 0.60)")
    parser.add_argument("--min-ocr-conf", type=float, default=0.35, help="Minimum OCR confidence threshold to log (default: 0.35)")
    parser.add_argument("--cooldown", type=float, default=3.0, help="Deduplication cooldown in seconds (default: 3.0)")
    args = parser.parse_args()

    # Load Ultralytics YOLO & EasyOCR
    try:
        from ultralytics import YOLO
        import easyocr
    except ImportError as e:
        print(f"[ERROR] Required package missing: {e}")
        print("Please run setup_env.bat and install_ocr.bat first!")
        sys.exit(1)

    base_dir = Path(__file__).parent.resolve()
    weights_path = Path(args.weights) if args.weights else base_dir / "runs" / "detect_plate" / "weights" / "best.pt"

    if not weights_path.exists():
        print(f"[ERROR] Model weights file not found: {weights_path}")
        print("Please run python train.py first to train the model!")
        sys.exit(1)

    print("===================================================")
    print("        ANPR System Pipeline (YOLO + OCR)         ")
    print("===================================================")
    print(f"Loading YOLO Model   : {weights_path}")
    model = YOLO(str(weights_path))

    print("Initializing EasyOCR Engine (English + Whitelist)...")
    reader = easyocr.Reader(['en'], gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # Setup CSV Output Logging
    results_dir = base_dir / "runs" / "anpr_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / "plates_log.csv"

    file_exists = csv_file.exists()
    csv_handle = open(csv_file, mode="a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_handle)

    if not file_exists:
        csv_writer.writerow(["Timestamp", "Plate_Number", "YOLO_Confidence", "OCR_Confidence", "Source_File"])
        csv_handle.flush()

    # Determine input source (file or camera index)
    source_val = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source_val)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        sys.exit(1)

    # Prepare Video Writer
    source_name = Path(args.source).stem if isinstance(source_val, str) else "live_cam"
    out_video_path = results_dir / f"anpr_{source_name}.mp4"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    print(f"Processing Video     : {args.source}")
    print(f"YOLO Conf Filter     : {args.conf}")
    print(f"Min OCR Conf Filter  : {args.min_ocr_conf}")
    print(f"Logging Results to   : {csv_file}")
    print(f"Output Video         : {out_video_path}")
    print("---------------------------------------------------")
    print("  Timestamp          | Plate Number | YOLO Conf | OCR Conf")
    print("---------------------------------------------------")

    seen_plates = {}  # {plate_str: last_seen_timestamp}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO plate detection
            results = model.predict(source=frame, conf=args.conf, verbose=False)
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                yolo_conf = float(box.conf[0])

                # Ensure valid bounding box crop coordinates within frame bounds
                h_f, w_f = frame.shape[:2]
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w_f, x2), min(h_f, y2)

                plate_crop = frame[y1_c:y2_c, x1_c:x2_c]
                if plate_crop.size == 0:
                    continue

                # Preprocess cropped plate image
                processed_crop = preprocess_plate_crop(plate_crop)

                # Run EasyOCR with character whitelist
                ocr_results = reader.readtext(processed_crop, allowlist=allowlist)

                for (bbox, text, ocr_conf) in ocr_results:
                    # Filter out low-confidence OCR guesses
                    if ocr_conf < args.min_ocr_conf:
                        continue

                    valid_plate = clean_and_validate_indian_plate(text)

                    if valid_plate:
                        curr_time = time.time()
                        last_seen = seen_plates.get(valid_plate, 0)

                        # Deduplication: Check cooldown period
                        if (curr_time - last_seen) > args.cooldown:
                            seen_plates[valid_plate] = curr_time
                            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            # Print to Terminal
                            print(f"  {timestamp_str} | {valid_plate:12} | {yolo_conf:.2f}      | {ocr_conf:.2f}")

                            # Write to CSV
                            csv_writer.writerow([timestamp_str, valid_plate, f"{yolo_conf:.2f}", f"{ocr_conf:.2f}", source_name])
                            csv_handle.flush()

                        # Draw bounding box and label on video frame
                        label = f"{valid_plate} ({ocr_conf:.0%})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Text background box
                        (w_lbl, h_lbl), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x1, y1 - h_lbl - 10), (x1 + w_lbl, y1), (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            out_video.write(frame)

    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user.")

    finally:
        cap.release()
        out_video.release()
        csv_handle.close()

    print("===================================================")
    print(" Processing Complete!")
    print(f" Plates Log CSV : {csv_file}")
    print(f" Output Video   : {out_video_path}")
    print("===================================================")

if __name__ == "__main__":
    run_anpr_pipeline()
