import os
from pathlib import Path
import cv2
import numpy as np
from .utils import bgr2nv12, letterbox_resize

class PlateDetector:
    """YOLOv8n License Plate Detector for RDK X5 (BPU-accelerated .bin)."""

    def __init__(self, model_path: str = None, conf_thresh: float = 0.60, iou_thresh: float = 0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.bpu_model = None
        self.fallback_model = None
        self.is_bpu = False
        
        base_dir = Path(__file__).resolve().parent.parent
        if model_path is None:
            # Default to BPU .bin model
            default_bin = base_dir / "models" / "yolov8n_plate_bayese_640x640_nv12.bin"
            default_onnx = base_dir.parent / "runs" / "detect_plate" / "weights" / "best.pt"
            self.model_path = default_bin if default_bin.exists() else default_onnx
        else:
            self.model_path = Path(model_path)

        self._load_model()

    def _load_model(self):
        """Load BPU hbm_runtime model, with fallback for local testing."""
        if str(self.model_path).endswith(".bin"):
            try:
                from hbm_runtime import HB_HBMRuntime
                print(f"[DETECTOR] Loading BPU Model on RDK X5: {self.model_path}")
                self.bpu_model = HB_HBMRuntime(str(self.model_path))
                self.is_bpu = True
                print("[DETECTOR] BPU Model loaded successfully (~5ms inference ready).")
                return
            except Exception as e:
                print(f"[WARN] Failed to load hbm_runtime BPU model: {e}")

        # Fallback to PyTorch/Ultralytics (for development/testing on laptop)
        try:
            from ultralytics import YOLO
            print(f"[DETECTOR] Loading standard YOLO fallback model: {self.model_path}")
            self.fallback_model = YOLO(str(self.model_path))
            self.is_bpu = False
            print("[DETECTOR] Fallback YOLO model loaded.")
        except Exception as e:
            print(f"[ERROR] Could not load any detector model: {e}")

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """
        Run plate detection on a frame.
        Returns:
            List of (x1, y1, x2, y2, confidence)
        """
        if frame is None or frame.size == 0:
            return []

        h_orig, w_orig = frame.shape[:2]

        if self.is_bpu and self.bpu_model:
            return self._detect_bpu(frame, h_orig, w_orig)
        elif self.fallback_model:
            return self._detect_fallback(frame)
        return []

    def _detect_bpu(self, frame: np.ndarray, h_orig: int, w_orig: int) -> list[tuple[int, int, int, int, float]]:
        """Execute detection on Horizon BPU using NV12 input format."""
        # 1. Letterbox resize to 640x640
        resized, scale, (dx, dy) = letterbox_resize(frame, (640, 640))
        # 2. Convert to NV12
        nv12_input = bgr2nv12(resized)
        # 3. BPU Forward pass
        outputs = self.bpu_model.run(nv12_input)
        
        # 4. Parse BPU output tensors & apply NMS
        # BPU output gives raw predictions (1, 5, 8400) or similar
        boxes = []
        if len(outputs) > 0:
            preds = outputs[0]
            if len(preds.shape) == 3 and preds.shape[1] < preds.shape[2]:
                preds = np.transpose(preds[0], (1, 0))  # shape: (8400, 5)
            elif len(preds.shape) == 3:
                preds = preds[0]

            for row in preds:
                conf = float(row[4]) if len(row) > 4 else 0.0
                if conf >= self.conf_thresh:
                    cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                    # Map back from letterbox to original frame coordinates
                    x1 = int(max(0, (cx - bw / 2 - dx) / scale))
                    y1 = int(max(0, (cy - bh / 2 - dy) / scale))
                    x2 = int(min(w_orig, (cx + bw / 2 - dx) / scale))
                    y2 = int(min(h_orig, (cy + bh / 2 - dy) / scale))
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2, conf))

        return self._apply_nms(boxes)

    def _detect_fallback(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Fallback detection using Ultralytics."""
        results = self.fallback_model.predict(source=frame, conf=self.conf_thresh, verbose=False)
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, conf))
        return boxes

    def _apply_nms(self, boxes: list[tuple[int, int, int, int, float]]) -> list[tuple[int, int, int, int, float]]:
        """Non-Maximum Suppression to remove overlapping duplicate boxes."""
        if not boxes:
            return []
            
        rects = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _ in boxes]
        scores = [conf for _, _, _, _, conf in boxes]
        
        indices = cv2.dnn.NMSBoxes(rects, scores, self.conf_thresh, self.iou_thresh)
        if len(indices) == 0:
            return []
            
        return [boxes[i] for i in indices.flatten()]
