import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from app.preprocessing.preprocess import run_pipeline

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)


def run_trocr(img):
    pil_img = Image.fromarray(img).convert("RGB")

    pixel_values = processor(
        images=pil_img,
        return_tensors="pt"
    ).pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    text = text.strip()

    confidence = 85 if text else 0

    return {
        "text": text,
        "words": [],
        "average_confidence": confidence,
        "engine": "trocr",
        "language": "eng",
    }

def split_into_lines(img):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    horizontal_projection = np.sum(thresh, axis=1)

    lines = []
    in_line = False
    start = 0

    for i, val in enumerate(horizontal_projection):
        if val > 10 and not in_line:
            start = i
            in_line = True
        elif val <= 10 and in_line:
            end = i
            if end - start > 15:
                lines.append((start, end))
            in_line = False

    cropped = []

    for (y1, y2) in lines:
        line_img = img[max(0, y1-5):min(img.shape[0], y2+5), :]
        cropped.append(line_img)

    return cropped

def clean_ocr_text(text):
    import re

    # remove repeated long numbers
    text = re.sub(r'\b\d{3,}\b', '', text)

    # remove decimal junk like 3pm.000 -> 3pm
    text = re.sub(r'(\w+)\.0+\b', r'\1', text)

    # remove repeated same number
    text = re.sub(r'\b(\d+)\s+\1\b', r'\1', text)

    # remove isolated symbols
    text = re.sub(r'[^\w\s\.,:\-\n]', '', text)

    # collapse extra spaces
    text = re.sub(r'[ ]{2,}', ' ', text)

    # clean spaces before punctuation
    text = re.sub(r'\s+([.,:])', r'\1', text)

    # split lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    cleaned_lines = []

    for i, line in enumerate(lines):
        line = line.strip()

        # remove suspicious trailing small numbers on titles
        if not line[:2].isdigit():
            line = re.sub(r'\s+\d{1,2}$', '', line)

        # capitalize first letter
        if line:
            line = line[0].upper() + line[1:]

        # normalize PM / AM
        line = re.sub(r'\b(\d+)\s*pm\b', r'\1 PM', line, flags=re.I)
        line = re.sub(r'\b(\d+)\s*am\b', r'\1 AM', line, flags=re.I)

        # add period if normal sentence
        if line and not re.match(r'^\d+\.', line):
            if line[-1].isalnum():
                line += '.'

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
 

def extract_text(source, save_steps=False):
    import cv2
    import numpy as np

    if isinstance(source, np.ndarray):
        img = source
    else:
        img = cv2.imread(str(source))

    if img is None:
        raise ValueError("Could not load image")

    img = cv2.resize(img, None, fx=1.5, fy=1.5)

    line_images = split_into_lines(img)

    all_text = []

    for line in line_images:
        result = run_trocr(line)
        txt = result["text"].strip()

        if txt:
            all_text.append(txt)

    final_text = "\n".join(all_text)
    final_text = clean_ocr_text(final_text)

    return {
        "text": final_text,
        "raw_text": final_text,
        "words": [],
        "average_confidence": 80 if final_text else 0,
        "word_count": len(final_text.split()),
        "engine": "trocr-lines",
        "language": "eng",
        "skew_corrected": 0
    }