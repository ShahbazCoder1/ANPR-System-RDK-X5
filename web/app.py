import os
import time
from pathlib import Path
import cv2
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

    def generate_mjpeg_stream():
        global current_frame_jpeg
        while True:
            if current_frame_jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_frame_jpeg + b'\r\n')
            time.sleep(0.033)  # ~30 FPS streaming

    @app.route("/api/feed")
    def video_feed():
        return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    return app

def update_live_frame(frame_bgr, fps: float = 0.0):
    """Update the global MJPEG frame buffer from the main detection loop."""
    global current_frame_jpeg, fps_counter
    if frame_bgr is not None:
        ret, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            current_frame_jpeg = buffer.tobytes()
            fps_counter = fps
