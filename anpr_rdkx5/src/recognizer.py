import cv2
import numpy as np
try:
    from src.plate_validator import clean_and_validate_plate
except ImportError:
    from .plate_validator import clean_and_validate_plate

class PlateRecognizer:
    """PaddleOCR Plate Text Recognizer with multi-version preprocessing."""

    def __init__(self, use_gpu: bool = False):
        self.ocr = None
        self._init_engine(use_gpu)

    def _init_engine(self, use_gpu: bool):
        """Initialize PaddleOCR engine."""
        try:
            from paddleocr import PaddleOCR
            print("[OCR] Initializing PaddleOCR Engine...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=use_gpu, enable_mkldnn=False)
            print("[OCR] PaddleOCR Engine initialized.")
        except Exception as e:
            print(f"[WARN] PaddleOCR initialization note: {e}")

    def preprocess_crop(self, crop_img: np.ndarray) -> list[np.ndarray]:
        """Return 3 enhanced versions of the plate crop (Upscaled, CLAHE, Sharpened)."""
        if crop_img is None or crop_img.size == 0:
            return []

        # 1. Add a 15% padding border so edge characters aren't clipped
        padding = max(8, int(crop_img.shape[0] * 0.15))
        padded = cv2.copyMakeBorder(crop_img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

        results = []
        h, w = padded.shape[:2]

        # Version 1: Upscaled Color Image
        if w > 0:
            scale = max(1.0, 350.0 / w)
            upscaled = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            results.append(upscaled)

            # Version 2: Grayscale + CLAHE Contrast Enhancement
            gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
            gray_upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_upscaled)
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            results.append(enhanced_bgr)

            # Version 3: Sharpened (effective on motion-blurred plates)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(upscaled, -1, kernel)
            results.append(sharpened)

        return results

    def recognize(self, plate_crop: np.ndarray) -> tuple[str | None, float, str]:
        """
        Run OCR on plate crop with multi-version preprocessing and validation.
        Returns:
            best_valid_plate (str | None),
            confidence (float),
            raw_text (str)
        """
        if self.ocr is None or plate_crop is None or plate_crop.size == 0:
            return None, 0.0, ""

        crop_versions = self.preprocess_crop(plate_crop)
        best_plate = None
        best_conf = 0.0
        best_raw = ""

        for crop_ver in crop_versions:
            try:
                res = self.ocr.ocr(crop_ver, cls=True)
            except Exception:
                continue

            if res and res[0]:
                texts = []
                confs = []
                for line in res[0]:
                    if len(line) >= 2 and len(line[1]) >= 2:
                        txt, score = line[1][0], float(line[1][1])
                        texts.append(txt)
                        confs.append(score)

                        # Validate individual text line
                        valid = clean_and_validate_plate(txt)
                        if valid and score > best_conf:
                            best_plate = valid
                            best_conf = score
                            best_raw = txt

                # Validate multi-line concatenated text
                if len(texts) > 1:
                    concat_txt = "".join(texts)
                    avg_score = sum(confs) / len(confs)
                    valid_concat = clean_and_validate_plate(concat_txt)
                    if valid_concat and avg_score > best_conf:
                        best_plate = valid_concat
                        best_conf = avg_score
                        best_raw = concat_txt

        return best_plate, best_conf, best_raw
