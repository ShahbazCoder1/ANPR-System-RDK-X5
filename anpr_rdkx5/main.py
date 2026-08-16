import argparse
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.camera import CameraStream
from src.database import TollDatabase
from src.detector import PlateDetector
from src.recognizer import PlateRecognizer
from src.utils import draw_plate_box
from web.app import create_app, update_live_frame

def parse_args():
    parser = argparse.ArgumentParser(description="RDK X5 Edge ANPR Toll Booth System")
    parser.add_argument("--source", type=str, default="mipi", 
                        help="Camera source: 'mipi' for GS130W, '0' for USB cam, or path to video file (e.g. test.mp4)")
    parser.add_argument("--model", type=str, default=None, 
                        help="Path to BPU .bin detector model")
    parser.add_argument("--conf", type=float, default=0.60, 
                        help="YOLO detection confidence threshold (default: 0.60)")
    parser.add_argument("--cooldown", type=float, default=10.0, 
                        help="Plate deduplication cooldown in seconds (default: 10.0)")
    parser.add_argument("--toll", type=int, default=100, 
                        help="Simulated toll fee in INR (default: 100)")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Flask server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, 
                        help="Flask server port (default: 5000)")
    parser.add_argument("--no-loop", action="store_true", 
                        help="Do not loop video files continuously")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable verbose debug logs and timings")
    return parser.parse_args()

def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    crops_dir = base_dir / "data" / "plate_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================")
    print("      RDK X5 ANPR Toll Booth System (BPU Accelerated)    ")
    print("=========================================================")
    print(f" Camera Source     : {args.source}")
    print(f" YOLO Conf Thresh  : {args.conf}")
    print(f" Toll Tariff       : ₹{args.toll} per vehicle")
    print(f" Dedup Cooldown    : {args.cooldown}s")
    print(f" Web Dashboard     : http://{args.host}:{args.port}")
    print("=========================================================")

    # 1. Initialize SQLite Database
    db = TollDatabase()
    print("[INIT] Database ready: data/anpr.db")

    # 2. Start Web Server in Background Thread
    app = create_app(db)
    flask_thread = threading.Thread(
        target=lambda: app.run(host=args.host, port=args.port, debug=False, use_reloader=False),
        daemon=True
    )
    flask_thread.start()
    print(f"[INIT] Web dashboard running at http://localhost:{args.port}")

    # 3. Initialize Detector & Recognizer
    detector = PlateDetector(model_path=args.model, conf_thresh=args.conf)
    recognizer = PlateRecognizer(use_gpu=False)

    # 4. Initialize Camera Stream
    camera = CameraStream(source=args.source, loop_video=not args.no_loop)

    print("\n---------------------------------------------------------")
    print("  Time     | Plate Number | Toll | YOLO  | OCR   | Source")
    print("---------------------------------------------------------")

    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    try:
        while True:
            t0 = time.time()
            ret, frame, ts = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % 15 == 0:
                elapsed = time.time() - fps_start_time
                current_fps = 15.0 / elapsed if elapsed > 0 else 0.0
                fps_start_time = time.time()

            h_f, w_f = frame.shape[:2]

            # Run Plate Detection (BPU)
            boxes = detector.detect(frame)

            for box in boxes:
                x1, y1, x2, y2, yolo_conf = box
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w_f, x2), min(h_f, y2)

                plate_crop = frame[y1_c:y2_c, x1_c:x2_c]
                if plate_crop.size == 0:
                    continue

                # Run OCR Recognition & Validation
                best_plate, ocr_conf, raw_text = recognizer.recognize(plate_crop)

                if best_plate:
                    # Check deduplication
                    is_dup = db.is_duplicate(best_plate, current_time_sec=ts, cooldown_sec=args.cooldown)
                    
                    if not is_dup:
                        # Save plate crop image
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                        crop_filename = f"crop_{now_str}_{best_plate}.jpg"
                        crop_path = crops_dir / crop_filename
                        cv2.imwrite(str(crop_path), plate_crop)

                        # Insert toll record into SQLite
                        source_tag = Path(args.source).stem if not str(args.source).isdigit() else "cam"
                        db.insert_record(
                            plate_number=best_plate,
                            yolo_conf=yolo_conf,
                            ocr_conf=ocr_conf,
                            raw_text=raw_text,
                            crop_filename=crop_filename,
                            source=source_tag,
                            toll_amount=args.toll,
                            current_time_sec=ts
                        )

                        time_display = datetime.now().strftime("%H:%M:%S")
                        print(f" {time_display} | {best_plate:12} | ₹{args.toll} | {yolo_conf:.0%}  | {ocr_conf:.0%}  | {source_tag}")

                    # Draw green bounding box & plate tag
                    draw_plate_box(frame, (x1, y1, x2, y2), best_plate, ocr_conf)
                else:
                    # Draw yellow box if detected by YOLO but OCR pending/unparsed
                    draw_plate_box(frame, (x1, y1, x2, y2), None, yolo_conf)

            # Update live stream buffer for web dashboard
            update_live_frame(frame, current_fps)

            if args.debug and frame_count % 30 == 0:
                dt = (time.time() - t0) * 1000
                print(f"[DEBUG] Frame latency: {dt:.1f}ms | BPU FPS: {current_fps:.1f}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopping ANPR System...")
    finally:
        camera.release()
        print("[INFO] Clean shutdown complete.")

if __name__ == "__main__":
    main()
