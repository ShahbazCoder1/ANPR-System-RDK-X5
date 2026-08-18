import cv2
import numpy as np
try:
    from src.plate_validator import clean_and_validate_plate
except ImportError:
    from .plate_validator import clean_and_validate_plate

class PlateRecognizer:
    """OCR Plate Text Recognizer with multi-engine support.
    
    Tries engines in order:
    1. RapidOCR (ONNX Runtime backend - no PaddlePaddle conflicts on RDK X5)
    2. PaddleOCR (full PaddlePaddle - works on laptop, conflicts on RDK X5)
    3. EasyOCR (PyTorch backend - fallback)
    """

    def __init__(self, use_gpu: bool = False):
        self.ocr = None
        self.engine_type = None  # "rapidocr", "paddleocr", or "easyocr"
        self._init_engine(use_gpu)

    def _init_engine(self, use_gpu: bool):
        """Initialize OCR engine with automatic fallback chain."""

        # 1. Try RapidOCR first (best for RDK X5 — uses ONNX Runtime, no PaddlePaddle)
        try:
            from rapidocr_onnxruntime import RapidOCR
            print("[OCR] Initializing RapidOCR (ONNX Runtime backend)...")
            self.ocr = RapidOCR()
            self.engine_type = "rapidocr"
            print("[OCR] RapidOCR Engine initialized successfully.")
            return
        except ImportError:
            print("[OCR] RapidOCR not installed, trying PaddleOCR...")
        except Exception as e:
            print(f"[OCR] RapidOCR init failed: {e}, trying PaddleOCR...")

        # 2. Try PaddleOCR (works on laptop, but conflicts with hobot_dnn on RDK X5)
        try:
            from paddleocr import PaddleOCR
            print("[OCR] Initializing PaddleOCR Engine...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, 
                                 use_gpu=use_gpu, enable_mkldnn=False)
            self.engine_type = "paddleocr"
            print("[OCR] PaddleOCR Engine initialized successfully.")
            return
        except Exception as e:
            print(f"[OCR] PaddleOCR init failed: {e}, trying EasyOCR...")

        # 3. Try EasyOCR (PyTorch fallback)
        try:
            import easyocr
            print("[OCR] Initializing EasyOCR (PyTorch backend)...")
            self.ocr = easyocr.Reader(['en'], gpu=use_gpu)
            self.engine_type = "easyocr"
            print("[OCR] EasyOCR Engine initialized successfully.")
            return
        except Exception as e:
            print(f"[ERROR] No OCR engine available: {e}")
            print("[ERROR] Install one: pip3 install rapidocr-onnxruntime")

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

        return results

    def _run_ocr(self, image: np.ndarray) -> list[tuple[str, float]]:
        """Run OCR on a single image, returns list of (text, confidence) tuples."""
        results = []

        if self.engine_type == "rapidocr":
            res = self.ocr(image)
            if res and res[0]:
                for line in res[0]:
                    # RapidOCR format: [box_coords, text, confidence]
                    if len(line) >= 3:
                        txt, score = str(line[1]), float(line[2])
                        results.append((txt, score))

        elif self.engine_type == "paddleocr":
            res = self.ocr.ocr(image, cls=True)
            if res and res[0]:
                for line in res[0]:
                    if len(line) >= 2 and len(line[1]) >= 2:
                        txt, score = line[1][0], float(line[1][1])
                        results.append((txt, score))

        elif self.engine_type == "easyocr":
            res = self.ocr.readtext(image)
            if res:
                for line in res:
                    if len(line) >= 3:
                        txt, score = str(line[1]), float(line[2])
                        results.append((txt, score))

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
                ocr_results = self._run_ocr(crop_ver)
            except Exception:
                continue

            if ocr_results:
                texts = []
                confs = []
                for txt, score in ocr_results:
                    texts.append(txt)
                    confs.append(score)

                    # Track best raw text regardless of validation
                    if score > best_conf and not best_plate:
                        best_raw = txt
                        
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
                    
                    if avg_score > best_conf and not best_plate:
                        best_raw = concat_txt

                    valid_concat = clean_and_validate_plate(concat_txt)
                    if valid_concat and avg_score > best_conf:
                        best_plate = valid_concat
                        best_conf = avg_score
                        best_raw = concat_txt

        return best_plate, best_conf, best_raw
