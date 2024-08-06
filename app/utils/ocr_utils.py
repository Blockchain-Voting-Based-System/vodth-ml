import logging
from paddleocr import PaddleOCR, draw_ocr
from app.settings import FONT_PATH

# Initialize PaddleOCR with English language support
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def apply_ocr_and_extract_text(img):
    result = ocr.ocr(img, cls=True)
    if result is None or len(result) == 0:
        logging.debug("No text detected by OCR.")
        return []
    texts = [word[1][0] for line in result if line is not None for word in line if word is not None]
    return texts
