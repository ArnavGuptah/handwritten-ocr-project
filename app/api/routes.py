import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.ocr.ocr_engine import extract_text
from app.nlp.nlp_processor import process

router = APIRouter()


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Only image files allowed.")
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read image.")
    ocr_result = extract_text(img)
    if not ocr_result["text"]:
        raise HTTPException(status_code=422, detail="No text found.")
    nlp_result = process(ocr_result["text"])
    return JSONResponse(content={
        "filename": file.filename,
        "ocr": {
        "raw_text": ocr_result["text"],
        "average_confidence": ocr_result["average_confidence"],
        "word_count": ocr_result.get("word_count", len(ocr_result.get("words", []))),
        "skew_corrected": ocr_result.get("skew_corrected", 0),
},
        "nlp": {
            "corrected_text": nlp_result["corrected_text"],
            "entities": nlp_result["entities"],
            "keywords": nlp_result["keywords"],
            "word_count": nlp_result["word_count"],
        },
        "status": "success",
    })


@router.get("/health")
async def health():
    return {"status": "ok"}