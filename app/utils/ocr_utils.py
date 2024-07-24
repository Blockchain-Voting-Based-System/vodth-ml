from paddleocr import PaddleOCR, draw_ocr
from app.settings import FONT_PATH

# Initialize PaddleOCR with English language support
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def apply_ocr_and_extract_text(img):
    result = ocr.ocr(img, cls=True)
    texts = [word[1][0] for line in result for word in line]
    return texts
