import difflib
import cv2
import numpy as np

def bgr2nv12(bgr_img: np.ndarray) -> np.ndarray:
    """Convert BGR image to NV12 format for Horizon BPU hardware models."""
    h, w = bgr_img.shape[:2]
    # Height and width must be even numbers
    if h % 2 != 0:
        bgr_img = bgr_img[:h-1, :]
        h -= 1
    if w % 2 != 0:
        bgr_img = bgr_img[:, :w-1]
        w -= 1
        
    yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420)
    y = yuv420p[:h, :]
    u = yuv420p[h : h + h // 4, :].reshape((h // 2, w // 2))
    v = yuv420p[h + h // 4 :, :].reshape((h // 2, w // 2))
    uv = np.empty((h // 2, w), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    nv12 = np.vstack((y, uv))
    return nv12

def nv122bgr(nv12_img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert NV12 format from MIPI camera/BPU back to BGR for display/saving."""
    return cv2.cvtColor(nv12_img, cv2.COLOR_YUV2BGR_NV12)

def letterbox_resize(image: np.ndarray, target_size=(640, 640)) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image with padding to maintain aspect ratio for YOLO."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized
    
    return canvas, scale, (dx, dy)

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate string similarity ratio using difflib SequenceMatcher."""
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def draw_plate_box(frame: np.ndarray, box: tuple[int, int, int, int], plate_text: str | None, conf: float) -> np.ndarray:
    """Draw stylish bounding box and plate label on frame for dashboard feed."""
    x1, y1, x2, y2 = box
    
    if plate_text:
        # Green box for recognized valid plate
        box_color = (0, 220, 0)
        label = f"{plate_text} ({conf:.0%})"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        
        # Label background pill
        cv2.rectangle(frame, (x1, max(0, y1 - h_lbl - 10)), (x1 + w_lbl + 8, y1), box_color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    else:
        # Yellow box for detected plate undergoing OCR
        box_color = (0, 215, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
    return frame
