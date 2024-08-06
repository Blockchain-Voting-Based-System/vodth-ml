from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import cv2
import io
from app.utils.image_processing import process_image, deskew
from app.utils.ocr_utils import apply_ocr_and_extract_text

app = FastAPI()

@app.post("/process_image/")
async def process_image_endpoint(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    roi = process_image(image)
    roi = deskew(roi)
    roi_th = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    M, N = roi_th.shape
    s = 5
    roi_th = roi_th[s: M - s, s: N - s]
    texts = apply_ocr_and_extract_text(roi_th)
    if len(texts) == 0:
        return JSONResponse(content={"error": "No text detected in the image."})
    return JSONResponse(content={"texts": texts})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
