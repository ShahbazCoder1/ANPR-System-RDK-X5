import os
from pathlib import Path
import cv2
import numpy as np
try:
    from src.utils import bgr2nv12, letterbox_resize
except ImportError:
    from .utils import bgr2nv12, letterbox_resize

class PlateDetector:
    """YOLOv8n License Plate Detector for RDK X5 (BPU-accelerated .bin)."""

    def __init__(self, model_path: str = None, conf_thresh: float = 0.60, iou_thresh: float = 0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.bpu_model = None
        self.fallback_model = None
        self.is_bpu = False
        self.use_parser = False  # True if hobot_dnn >= 2.6.1 with built-in YOLO parser
        self._debug_printed = False  # Print BPU output structure once
        
        base_dir = Path(__file__).resolve().parent.parent
        if model_path is None:
            default_bin = base_dir / "models" / "yolov8n_plate_bayese_640x640_nv12.bin"
            default_onnx = base_dir.parent / "runs" / "detect_plate" / "weights" / "best.pt"
            self.model_path = default_bin if default_bin.exists() else default_onnx
        else:
            self.model_path = Path(model_path)

        self._load_model()

    def _load_model(self):
        """Load BPU model via hobot_dnn (pyeasy_dnn), with fallback to Ultralytics."""
        if str(self.model_path).endswith(".bin"):
            if not self.model_path.exists():
                print(f"[ERROR] BPU model file not found: {self.model_path}")
                print("Please copy 'yolov8n_plate_bayese_640x640_nv12.bin' into your 'models/' folder!")
                return
            try:
                from hobot_dnn import pyeasy_dnn as dnn
                print(f"[DETECTOR] Loading BPU Model on RDK X5: {self.model_path}")
                
                # Try loading with built-in ultralytics_yolo parser (hobot_dnn >= 2.6.1)
                try:
                    models = dnn.load(str(self.model_path), parser="ultralytics_yolo")
                    self.bpu_model = models[0]
                    self.use_parser = True
                    print("[DETECTOR] BPU Model loaded with built-in YOLO parser (NMS on BPU).")
                except Exception:
                    # Fallback: load without parser, manual post-processing
                    models = dnn.load(str(self.model_path))
                    self.bpu_model = models[0]
                    self.use_parser = False
                    print("[DETECTOR] BPU Model loaded (manual post-processing mode).")
                
                self.is_bpu = True
                print("[DETECTOR] BPU Model loaded successfully (~5ms inference ready).")
                return
            except ImportError:
                print("[WARN] hobot_dnn not found. Not running on RDK X5 board.")
            except Exception as e:
                print(f"[ERROR] Failed to initialize BPU model: {e}")
                return

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
        """Execute detection on Horizon BPU via hobot_dnn."""
        # 1. Letterbox resize to 640x640
        resized, scale, (dx, dy) = letterbox_resize(frame, (640, 640))
        # 2. Convert to NV12 for BPU hardware input
        nv12_input = bgr2nv12(resized)

        # 3. BPU Forward pass
        outputs = self.bpu_model.forward([nv12_input])

        boxes = []

        if self.use_parser:
            # Built-in parser returns parsed results directly
            try:
                for det in outputs:
                    if hasattr(det, 'buffer'):
                        parsed = np.array(det.buffer, copy=False)
                    else:
                        parsed = np.array(det, copy=False)
                    # Parse each detection row: [x1, y1, x2, y2, score, class_id]
                    for row in parsed.reshape(-1, 6):
                        conf = float(row[4])
                        if conf >= self.conf_thresh:
                            x1 = int(max(0, (row[0] - dx) / scale))
                            y1 = int(max(0, (row[1] - dy) / scale))
                            x2 = int(min(w_orig, (row[2] - dx) / scale))
                            y2 = int(min(h_orig, (row[3] - dy) / scale))
                            if x2 > x1 and y2 > y1:
                                boxes.append((x1, y1, x2, y2, conf))
            except Exception as e:
                print(f"[WARN] Parser output parsing error: {e}, falling back to manual.")
                boxes = self._parse_raw_outputs(outputs, scale, dx, dy, h_orig, w_orig)
        else:
            # Manual post-processing: raw BPU output tensors
            boxes = self._parse_raw_outputs(outputs, scale, dx, dy, h_orig, w_orig)

        return self._apply_nms(boxes)

    def _parse_raw_outputs(self, outputs, scale, dx, dy, h_orig, w_orig):
        """Parse raw BPU output tensors manually (no built-in parser)."""
        boxes = []
        
        try:
            # Get the first output tensor's buffer as numpy
            if hasattr(outputs[0], 'buffer'):
                preds = np.array(outputs[0].buffer, copy=False).astype(np.float32)
            else:
                preds = np.array(outputs[0], copy=False).astype(np.float32)
            
            # Squeeze ALL size-1 dimensions: (1, 5, 8400, 1) → (5, 8400)
            preds = np.squeeze(preds)
            
            # YOLOv8 output is (5, 8400) for single-class: transpose to (8400, 5)
            if len(preds.shape) == 2 and preds.shape[0] < preds.shape[1]:
                preds = preds.T  # Now (8400, 5): each row = [cx, cy, w, h, conf]

            for row in preds:
                if len(row) < 5:
                    continue
                conf = float(row[4])
                if conf >= self.conf_thresh:
                    cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    x1 = int(max(0, (cx - bw / 2 - dx) / scale))
                    y1 = int(max(0, (cy - bh / 2 - dy) / scale))
                    x2 = int(min(w_orig, (cx + bw / 2 - dx) / scale))
                    y2 = int(min(h_orig, (cy + bh / 2 - dy) / scale))
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2, conf))
        except Exception as e:
            print(f"[WARN] Raw output parsing error: {e}")
            # Debug: print output structure to help diagnose
            for i, out in enumerate(outputs):
                if hasattr(out, 'buffer'):
                    arr = np.array(out.buffer, copy=False)
                    print(f"  Output[{i}]: shape={arr.shape}, dtype={arr.dtype}")
                elif hasattr(out, 'shape'):
                    print(f"  Output[{i}]: shape={out.shape}, dtype={out.dtype}")
                else:
                    print(f"  Output[{i}]: type={type(out)}")

        return boxes

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
