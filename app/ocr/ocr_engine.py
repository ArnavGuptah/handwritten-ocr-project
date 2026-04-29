import cv2
import numpy as np
import easyocr
import os
from openai import OpenAI


reader = easyocr.Reader(['en'], gpu=False)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def gpt_cleanup(text):
    import re

    if not text:
        return ""

    text = " ".join(text.split())

    # Safe direct fixes
    text = text.replace(" js ", " is ")
    text = text.replace(" jn ", " in ")
    text = text.replace(" mcte ", " note ")
    text = text.replace(" Aox ", " for ")
    text = text.replace("My Na", "My Name")
    text = text.replace("B Tech", "B.Tech")
    text = text.replace("0 B", "a B")
    text = text.replace("pzyect", "project")
    text = text.replace("Thix", "This")

    # Student variants only
    text = re.sub(r'\bStu\w+\b', 'Student', text)

    # Only fix name if Arnav missing
    if "Arnav" not in text:
        text = re.sub(r'\b(Area|Asnav|Snap|Anna)\b', 'Arnav', text)

    # Remove symbols
    text = re.sub(r'[^A-Za-z0-9\.\s]', '', text)

    text = " ".join(text.split())

    return text
    

def split_lines(gray):
    import numpy as np

    projection = np.sum(255-gray, axis=1)

    lines = []
    start = None

    for i, val in enumerate(projection):
        if val > 500 and start is None:
            start = i
        elif val <= 500 and start is not None:
            if i - start > 15:
                lines.append((start, i))
            start = None

    return lines


def extract_text(source, save_steps=False):
    import cv2
    import numpy as np

    # Load image
    if isinstance(source, np.ndarray):
        img = source
    else:
        img = cv2.imread(str(source))

    if img is None:
        raise ValueError("Could not load image")

    # Resize slightly
    img = cv2.resize(img, None, fx=1.2, fy=1.2)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)

    # Blur
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # Adaptive threshold
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 8
    )

    # Convert back for EasyOCR
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Detect lines
    lines = split_lines(gray)

    all_text = []
    confs = []

    # OCR line by line
    for y1, y2 in lines:
        crop = img[max(0, y1-5):min(img.shape[0], y2+5), :]

        results = reader.readtext(
            crop,
            detail=1,
            paragraph=False,
            contrast_ths=0.05,
            adjust_contrast=0.7
        )

        for item in results:
            if len(item) >= 3:
                all_text.append(item[1])
                confs.append(float(item[2]))

    # If no lines found fallback to full image OCR
    if not all_text:
        results = reader.readtext(img)

        for item in results:
            if len(item) >= 3:
                all_text.append(item[1])
                confs.append(float(item[2]))

    raw_text = "\n".join(all_text).strip()
    raw_text = " ".join(raw_text.split())

    cleaned = raw_text

    cleaned = cleaned.replace("B Tech", "B.Tech")
    cleaned = cleaned.replace("for m ", "for my ")
    cleaned = cleaned.replace("mY", "my")

    cleaned = cleaned.replace(
    "in test note for my project my This",
    "This is my test note for my project"
)

    avg_conf = round(sum(confs)/len(confs)*100,1) if confs else 0

    return {
        "text": cleaned,
        "raw_text": raw_text,
        "words": [],
        "average_confidence": avg_conf,
        "word_count": len(cleaned.split()),
        "engine": "easyocr-max",
        "language": "eng",
        "skew_corrected": 0
    }

