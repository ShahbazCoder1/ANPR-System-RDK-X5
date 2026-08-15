import sys
import time
from pathlib import Path
import cv2
import numpy as np

class CameraStream:
    """Unified Camera abstraction for RDK X5 (GS130W MIPI CSI, USB, and Video Files)."""

    def __init__(self, source: str = "mipi", width: int = 1280, height: int = 720, fps: int = 30, loop_video: bool = True):
        self.source = source
        self.target_w = width
        self.target_h = height
        self.fps = fps
        self.loop_video = loop_video
        self.is_mipi = str(source).lower().startswith("mipi")
        self.cam_obj = None
        self.cap = None
        self.open()

    def open(self):
        """Open the selected camera or video file source."""
        if self.is_mipi:
            try:
                # Try RDK X5 hardware MIPI VIO library
                try:
                    from hobot_vio import libsrcampy as srcampy
                except ImportError:
                    from hobot_vio_rdkx5 import libsrcampy as srcampy

                print(f"[CAMERA] Initializing GS130W MIPI CSI Camera ({self.target_w}x{self.target_h} @ {self.fps}fps)...")
                self.cam_obj = srcampy.Camera()
                # Open camera: (video_index 0, fps 30, [w1, w2], [h1, h2], sensor_h, sensor_w)
                ret = self.cam_obj.open_cam(0, -1, self.fps, [640, self.target_w], [640, self.target_h], self.target_h, self.target_w)
                if ret != 0:
                    print(f"[ERROR] Failed to open MIPI camera! Error code: {ret}")
                    self.cam_obj = None
                else:
                    print("[CAMERA] GS130W MIPI Camera opened successfully (Zero-Copy NV12 ready).")
            except Exception as e:
                print(f"[WARN] libsrcampy MIPI not available on host ({e}). Falling back to OpenCV.")
                self.is_mipi = False

        if not self.is_mipi:
            # OpenCV source: integer index for USB camera, or file path for prerecorded video
            source_val = int(self.source) if str(self.source).isdigit() else self.source
            print(f"[CAMERA] Opening OpenCV video source: {self.source}")
            self.cap = cv2.VideoCapture(source_val)
            if not self.cap.isOpened():
                print(f"[ERROR] Cannot open video source: {self.source}")
                sys.exit(1)
            print(f"[CAMERA] Video source ready: {self.source}")

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """
        Read next frame.
        Returns:
            success (bool),
            frame_bgr (np.ndarray),
            timestamp_sec (float)
        """
        if self.is_mipi and self.cam_obj:
            try:
                # Channel 1: High-res for display/web (format 2 = NV12)
                raw_nv12 = self.cam_obj.get_img(2, self.target_w, self.target_h)
                if raw_nv12:
                    nv12_arr = np.frombuffer(raw_nv12, dtype=np.uint8)
                    frame_bgr = cv2.cvtColor(nv12_arr.reshape((int(self.target_h * 1.5), self.target_w)), cv2.COLOR_YUV2BGR_NV12)
                    return True, frame_bgr, time.time()
            except Exception as e:
                print(f"[WARN] Error reading MIPI frame: {e}")
                return False, None, 0.0

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret and self.loop_video and not str(self.source).isdigit():
                # Rewind video for continuous toll demonstration
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                
            if ret:
                pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                ts = pos_msec / 1000.0 if pos_msec > 0 else time.time()
                return True, frame, ts
            return False, None, 0.0

        return False, None, 0.0

    def release(self):
        """Release camera and video capture resources."""
        if self.cam_obj:
            try:
                self.cam_obj.close_cam()
                print("[CAMERA] MIPI Camera closed.")
            except Exception:
                pass
        if self.cap:
            self.cap.release()
            print("[CAMERA] Video capture released.")
