import argparse
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from collections import deque
import cv2
import numpy as np

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
    parser.add_argument("--skip", type=int, default=2,
                        help="Process every Nth frame for detection (default: 2). Higher = faster but may miss plates.")
    return parser.parse_args()


# ---------- Background OCR Worker ----------
# YOLO runs on every processed frame (fast BPU ~5ms),
# but OCR runs in a background thread so it doesn't block the video stream.

class OCRWorker:
    """Background OCR worker thread — processes plate crops without blocking video."""

    def __init__(self, recognizer, db, crops_dir, toll_amount, source_tag, cooldown):
        self.recognizer = recognizer
        self.db = db
        self.crops_dir = crops_dir
        self.toll_amount = toll_amount
        self.source_tag = source_tag
        self.cooldown = cooldown
        self._queue = deque(maxlen=20)  # Drop old crops if overwhelmed
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # Recent results for drawing on frames (thread-safe)
        self.recent_plates = {}  # {plate_text: (x1,y1,x2,y2,conf,expire_time)}

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, plate_crop, box, yolo_conf, timestamp):
        """Submit a plate crop for background OCR processing."""
        with self._lock:
            self._queue.append((plate_crop.copy(), box, yolo_conf, timestamp))

    def _run(self):
        """OCR worker loop — processes one crop at a time."""
        while not self._stop.is_set():
            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.popleft()

            if item is None:
                time.sleep(0.01)
                continue

            plate_crop, box, yolo_conf, ts = item
            x1, y1, x2, y2 = box

            try:
                best_plate, ocr_conf, raw_text = self.recognizer.recognize(plate_crop)
            except Exception as e:
                print(f"[WARN] OCR error: {e}")
                continue

            # Debug: always print what OCR sees
            if raw_text:
                print(f"[OCR] YOLO={yolo_conf:.0%} | Raw='{raw_text}' | Validated='{best_plate or 'REJECTED'}'")
            
            # Determine the plate text to use (validated or raw fallback)
            import re
            display_plate = best_plate
            if not display_plate and raw_text:
                # Use cleaned raw text as fallback
                display_plate = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
                if len(display_plate) < 4:
                    display_plate = None

            if display_plate:
                is_dup = self.db.is_duplicate(display_plate, current_time_sec=ts, cooldown_sec=self.cooldown)
                if not is_dup:
                    # Save plate crop image
                    now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                    crop_filename = f"crop_{now_str}_{display_plate}.jpg"
                    crop_path = self.crops_dir / crop_filename
                    cv2.imwrite(str(crop_path), plate_crop)

                    # Insert toll record into SQLite
                    self.db.insert_record(
                        plate_number=display_plate,
                        yolo_conf=yolo_conf,
                        ocr_conf=ocr_conf,
                        raw_text=raw_text,
                        crop_filename=crop_filename,
                        source=self.source_tag,
                        toll_amount=self.toll_amount,
                        current_time_sec=ts
                    )

                    tag = "" if best_plate else " (raw)"
                    time_display = datetime.now().strftime("%H:%M:%S")
                    print(f" {time_display} | {display_plate:12} | ₹{self.toll_amount} | {yolo_conf:.0%}  | {ocr_conf:.0%}  | {self.source_tag}{tag}")

                # Cache for drawing green box on subsequent frames
                self.recent_plates[display_plate] = (x1, y1, x2, y2, ocr_conf, time.time() + 3.0)

    def stop(self):
        self._stop.set()


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
    print(f" Frame Skip        : every {args.skip} frame(s)")
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

    # 5. Start Background OCR Worker
    source_tag = Path(args.source).stem if not str(args.source).isdigit() else "cam"
    ocr_worker = OCRWorker(recognizer, db, crops_dir, args.toll, source_tag, args.cooldown)

    print("\n---------------------------------------------------------")
    print("  Time     | Plate Number | Toll | YOLO  | OCR   | Source")
    print("---------------------------------------------------------")

    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    try:
        while True:
            t0 = time.time()
            ret, frame, ts = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_frame_count / elapsed
                fps_start_time = time.time()
                fps_frame_count = 0

            h_f, w_f = frame.shape[:2]

            # Run YOLO detection only every N frames (BPU is fast, but saves CPU for OCR)
            if frame_count % args.skip == 0:
                boxes = detector.detect(frame)

                for box in boxes:
                    x1, y1, x2, y2, yolo_conf = box
                    x1_c, y1_c = max(0, x1), max(0, y1)
                    x2_c, y2_c = min(w_f, x2), min(h_f, y2)

                    plate_crop = frame[y1_c:y2_c, x1_c:x2_c]
                    if plate_crop.size == 0:
                        continue

                    # Draw yellow detection box immediately (fast)
                    draw_plate_box(frame, (x1, y1, x2, y2), None, yolo_conf)

                    # Submit to background OCR worker (non-blocking)
                    ocr_worker.submit(plate_crop, (x1, y1, x2, y2), yolo_conf, ts)

            # Overlay any recently recognized plates (green boxes from OCR worker)
            now = time.time()
            expired = []
            for plate, (px1, py1, px2, py2, pconf, expire) in ocr_worker.recent_plates.items():
                if now < expire:
                    draw_plate_box(frame, (px1, py1, px2, py2), plate, pconf)
                else:
                    expired.append(plate)
            for plate in expired:
                del ocr_worker.recent_plates[plate]

            # Update live stream buffer for web dashboard (EVERY frame — smooth video)
            update_live_frame(frame, current_fps)

            if args.debug and frame_count % 30 == 0:
                dt = (time.time() - t0) * 1000
                print(f"[DEBUG] Frame latency: {dt:.1f}ms | Pipeline FPS: {current_fps:.1f}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopping ANPR System...")
    finally:
        ocr_worker.stop()
        camera.release()
        print("[INFO] Clean shutdown complete.")

if __name__ == "__main__":
    main()
