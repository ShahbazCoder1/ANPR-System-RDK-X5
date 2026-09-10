import os
import time
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, send_from_directory
from src.database import TollDatabase

# Global frame buffer for live MJPEG video stream
current_frame_jpeg = None
fps_counter = 0.0

def create_app(db: TollDatabase = None) -> Flask:
    base_dir = Path(__file__).resolve().parent.parent
    template_dir = base_dir / "web" / "templates"
    static_dir = base_dir / "web" / "static"
    crops_dir = base_dir / "data" / "plate_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
    app.config["DB"] = db if db is not None else TollDatabase()
    app.config["CROPS_DIR"] = crops_dir
    app.config["START_TIME"] = time.time()

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/api/stats")
    def get_stats():
        db_instance: TollDatabase = app.config["DB"]
        stats = db_instance.get_stats()
        
        # Calculate uptime
        uptime_sec = int(time.time() - app.config["START_TIME"])
        hours, remainder = divmod(uptime_sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        stats["uptime"] = f"{hours}h {minutes}m {seconds}s"
        stats["fps"] = round(fps_counter, 1)
        
        return jsonify(stats)

    @app.route("/api/recent")
    def get_recent():
        db_instance: TollDatabase = app.config["DB"]
        records = db_instance.get_recent(limit=20)
        return jsonify(records)

    @app.route("/api/crops/<path:filename>")
    def get_crop_image(filename):
        return send_from_directory(str(app.config["CROPS_DIR"]), filename)

    @app.route("/api/snapshot")
    def get_snapshot():
        """Single JPEG snapshot endpoint for ultra-low bandwidth / lightweight browser rendering."""
        global current_frame_jpeg
        if current_frame_jpeg is not None:
            return Response(current_frame_jpeg, mimetype='image/jpeg')
        return Response(b'', status=204)

    def generate_mjpeg_stream():
        global current_frame_jpeg
        placeholder_jpeg = None
        
        while True:
            frame_bytes = current_frame_jpeg
            if frame_bytes is None:
                if placeholder_jpeg is None:
                    # Create lightweight 640x360 placeholder while camera initializes
                    placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "RDK X5 ANPR Initializing...", (130, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (56, 189, 248), 2)
                    _, buf = cv2.imencode('.jpg', placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                    placeholder_jpeg = buf.tobytes()
                frame_bytes = placeholder_jpeg

            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except (GeneratorExit, Exception):
                break
                
            time.sleep(0.066)  # ~15 FPS: perfectly smooth for humans, ultra-low CPU on embedded board

    @app.route("/api/feed")
    def video_feed():
        return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    return app

def update_live_frame(frame_bgr, fps: float = 0.0):
    """Update the global MJPEG frame buffer from the main detection loop with embedded-optimized sizing."""
    global current_frame_jpeg, fps_counter
    if frame_bgr is not None:
        h, w = frame_bgr.shape[:2]
        # Resize to max 640 width to protect embedded browser memory & CPU
        if w > 640:
            scale = 640.0 / w
            preview = cv2.resize(frame_bgr, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            preview = frame_bgr
            
        ret, buffer = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if ret:
            current_frame_jpeg = buffer.tobytes()
            fps_counter = fps
