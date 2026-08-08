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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Valid 2-letter State & UT Codes in India
VALID_INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB", "BH"
}

# Indian License Plate RegEx: State(2) + District(1-2) + Series(1-3) + Number(4)
INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")


def fix_state_code(char0: str, char1: str) -> tuple[str, str] | None:
    """Try to fix the first 2 characters into a valid Indian state code."""
    # Direct match
    code = char0 + char1
    if code in VALID_INDIAN_STATES:
        return char0, char1

    # Common OCR misreads for each character position
    char0_fixes = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '4': 'A', '6': 'G'}
    char1_fixes = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '4': 'A', '6': 'G',
                   'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'}

    # Try fixing char0 only
    c0 = char0_fixes.get(char0, char0)
    if (c0 + char1) in VALID_INDIAN_STATES:
        return c0, char1

    # Try fixing char1 only
    c1 = char1_fixes.get(char1, char1)
    if (char0 + c1) in VALID_INDIAN_STATES:
        return char0, c1

    # Try fixing both
    if (c0 + c1) in VALID_INDIAN_STATES:
        return c0, c1

    # Specific known misreads
    known_fixes = {
        "MJ": "MH", "MB": "MH", "ME": "MH", "M0": "MH", "M8": "MH",
        "W8": "WB", "VB": "WB", "V8": "WB", "WE": "WB", "W0": "WB",
        "IB": "WB", "1B": "WB",
        "D1": "DL", "D0": "DL", "OL": "DL",
        "K4": "KA", "K8": "KA", "KB": "KA",
        "Y0": "UP", "YP": "UP", "YE": "UP",
        "T5": "TS", "7N": "TN", "7S": "TS",
        "H8": "HR", "HB": "HR",
        "G1": "GJ", "0D": "OD",
    }
    if code in known_fixes:
        fixed = known_fixes[code]
        return fixed[0], fixed[1]

    return None


def clean_and_validate_plate(raw_text: str) -> str | None:
    """Clean OCR text, auto-correct confusions, validate Indian plate format."""
    if not raw_text:
        return None

    # Uppercase and strip non-alphanumeric
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())

    if len(cleaned) < 7 or len(cleaned) > 12:
        return None

    fixed = list(cleaned)

    # Fix state code (first 2 chars)
    state_fix = fix_state_code(fixed[0], fixed[1])
    if state_fix is None:
        return None
    fixed[0], fixed[1] = state_fix

    # Fix district digits (position 2-3 must be numbers)
    for i in range(2, min(4, len(fixed))):
        ch = fixed[i]
        if not ch.isdigit():
            digit_map = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'}
            if ch in digit_map:
                fixed[i] = digit_map[ch]

    # Fix last 4 chars (must be numbers)
    for i in range(max(4, len(fixed) - 4), len(fixed)):
        ch = fixed[i]
        if not ch.isdigit():
            digit_map = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'}
            if ch in digit_map:
                fixed[i] = digit_map[ch]

    result = "".join(fixed)

    # Validate final format
    if result[:2] in VALID_INDIAN_STATES and INDIAN_PLATE_REGEX.match(result):
        return result

    return None


def preprocess_plate_crop(crop_img: np.ndarray) -> list[np.ndarray]:
    """
    Return multiple preprocessed versions of the plate crop.
    EasyOCR will be run on each version and the best result is used.
    """
    if crop_img is None or crop_img.size == 0:
        return [crop_img]

    results = []

    # Version 1: Upscaled color crop (EasyOCR handles its own internal preprocessing)
    h, w = crop_img.shape[:2]
    if w > 0:
        scale = max(1.0, 350 / w)
        upscaled = cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        results.append(upscaled)

    # Version 2: Grayscale + CLAHE (enhanced contrast, no binarization)
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    if w > 0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    results.append(enhanced)

    # Version 3: Sharpened grayscale
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    results.append(sharpened)

    return results


def run_anpr_pipeline():
    parser = argparse.ArgumentParser(description="End-to-End ANPR Pipeline (YOLOv8 + EasyOCR)")
    parser.add_argument("--source", type=str, required=True, help="Path to video file or camera index (e.g. 0)")
    parser.add_argument("--weights", type=str, default="", help="Custom YOLO weights path")
    parser.add_argument("--conf", type=float, default=0.60, help="YOLO confidence threshold (default: 0.60)")
    parser.add_argument("--cooldown", type=float, default=3.0, help="Deduplication cooldown seconds (default: 3.0)")
    parser.add_argument("--debug", action="store_true", help="Save plate crops and raw OCR text for debugging")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
        import easyocr
        import torch
    except ImportError as e:
        print(f"[ERROR] Missing package: {e}")
        print("Run setup_env.bat and install_ocr.bat first!")
        sys.exit(1)

    base_dir = Path(__file__).parent.resolve()
    weights_path = Path(args.weights) if args.weights else base_dir / "runs" / "detect_plate" / "weights" / "best.pt"

    if not weights_path.exists():
        print(f"[ERROR] Model not found: {weights_path}")
        sys.exit(1)

    # Use torch to detect GPU properly (not cv2.cuda)
    use_gpu = torch.cuda.is_available()

    print("===================================================")
    print("        ANPR System Pipeline (YOLO + OCR)         ")
    print("===================================================")
    print(f"YOLO Model     : {weights_path}")
    model = YOLO(str(weights_path))

    print(f"EasyOCR GPU    : {use_gpu}")
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # Output dirs
    results_dir = base_dir / "runs" / "anpr_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        debug_dir = results_dir / "debug_crops"
        debug_dir.mkdir(parents=True, exist_ok=True)

    csv_file = results_dir / "plates_log.csv"
    file_exists = csv_file.exists()
    csv_handle = open(csv_file, mode="a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_handle)
    if not file_exists:
        csv_writer.writerow(["Timestamp", "Plate_Number", "YOLO_Confidence", "OCR_Confidence", "Raw_OCR_Text", "Source"])
        csv_handle.flush()

    source_val = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.source}")
        sys.exit(1)

    source_name = Path(args.source).stem if isinstance(source_val, str) else "live_cam"
    out_video_path = results_dir / f"anpr_{source_name}.mp4"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Video Source   : {args.source}")
    print(f"YOLO Conf      : {args.conf}")
    print(f"Output Video   : {out_video_path}")
    print(f"Debug Mode     : {args.debug}")
    print("---------------------------------------------------")
    print("  Timestamp          | Plate Number | YOLO  | OCR   | Raw OCR Text")
    print("---------------------------------------------------")

    seen_plates = {}
    frame_count = 0
    detection_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results = model.predict(source=frame, conf=args.conf, verbose=False)
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                yolo_conf = float(box.conf[0])

                h_f, w_f = frame.shape[:2]
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w_f, x2), min(h_f, y2)

                plate_crop = frame[y1_c:y2_c, x1_c:x2_c]
                if plate_crop.size == 0:
                    continue

                # Get multiple preprocessed versions
                crop_versions = preprocess_plate_crop(plate_crop)

                # Run OCR on each version, collect all results
                best_plate = None
                best_ocr_conf = 0.0
                best_raw_text = ""

                for crop_ver in crop_versions:
                    ocr_results = reader.readtext(crop_ver, allowlist=allowlist)

                    for (bbox, text, ocr_conf) in ocr_results:
                        raw_text = text.strip()

                        # Try to validate this OCR reading
                        valid_plate = clean_and_validate_plate(raw_text)

                        if valid_plate and ocr_conf > best_ocr_conf:
                            best_plate = valid_plate
                            best_ocr_conf = ocr_conf
                            best_raw_text = raw_text

                        # Also try concatenating all OCR text segments from this crop
                        # (plates sometimes get split into multiple text segments)
                    
                    # Try concatenating all text from this crop version
                    if ocr_results:
                        all_text = "".join([t for (_, t, _) in ocr_results])
                        avg_conf = sum([c for (_, _, c) in ocr_results]) / len(ocr_results)
                        valid_concat = clean_and_validate_plate(all_text)
                        if valid_concat and avg_conf > best_ocr_conf:
                            best_plate = valid_concat
                            best_ocr_conf = avg_conf
                            best_raw_text = all_text

                # Save debug crops
                if args.debug:
                    crop_path = debug_dir / f"frame{frame_count}_crop{detection_count}.jpg"
                    cv2.imwrite(str(crop_path), plate_crop)
                    detection_count += 1

                if best_plate:
                    curr_time = time.time()
                    last_seen = seen_plates.get(best_plate, 0)

                    if (curr_time - last_seen) > args.cooldown:
                        seen_plates[best_plate] = curr_time
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        print(f"  {ts} | {best_plate:12} | {yolo_conf:.2f}  | {best_ocr_conf:.2f}  | {best_raw_text}")

                        csv_writer.writerow([ts, best_plate, f"{yolo_conf:.2f}", f"{best_ocr_conf:.2f}", best_raw_text, source_name])
                        csv_handle.flush()

                    # Draw green box + plate text
                    label = f"{best_plate} ({best_ocr_conf:.0%})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - h_lbl - 10), (x1 + w_lbl, y1), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                else:
                    # Yellow box = detected plate but OCR failed to read valid text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            out_video.write(frame)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        out_video.release()
        csv_handle.close()

    print("===================================================")
    print(f" Processing Complete! ({frame_count} frames)")
    print(f" CSV Log    : {csv_file}")
    print(f" Video      : {out_video_path}")
    if args.debug:
        print(f" Debug Crops: {debug_dir}")
    print("===================================================")

if __name__ == "__main__":
    run_anpr_pipeline()
