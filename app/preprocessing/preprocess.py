import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io


def load_image(source):
    if isinstance(source, np.ndarray):
        return source.copy()
    if isinstance(source, Image.Image):
        arr = np.array(source.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    img = cv2.imread(str(source), cv2.IMREAD_COLOR)
    return img


def to_grayscale(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance_contrast(img, factor=1.4):
    pil_img = Image.fromarray(img)
    enhanced = ImageEnhance.Contrast(pil_img).enhance(factor)
    return np.array(enhanced)


def remove_noise(img, method="nlmeans", strength=10):
    if method == "nlmeans":
        return cv2.fastNlMeansDenoising(img, h=strength)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (3, 3), 0)
    if method == "median":
        return cv2.medianBlur(img, 3)


def apply_threshold(img, method="adaptive"):
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
    if method == "otsu":
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary


def deskew(img):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    best_angle = 0.0
    best_score = -1.0
    h, w = img.shape[:2]
    centre = (w // 2, h // 2)

    for angle in np.arange(-10, 10, 0.5):
        M = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(edges, M, (w, h))
        score = rotated.sum(axis=1).astype(np.float64).var()
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    if abs(best_angle) < 0.3:
        return img, 0.0

    M = cv2.getRotationMatrix2D(centre, best_angle, 1.0)
    corrected = cv2.warpAffine(img, M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=255)
    return corrected, best_angle


def resize_for_ocr(img, factor=1.5):
    if factor <= 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)),
                      interpolation=cv2.INTER_LANCZOS4)


def add_border(img, px=20):
    return cv2.copyMakeBorder(img, px, px, px, px,
                              cv2.BORDER_CONSTANT, value=255)

def remove_horizontal_lines(img):
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    inverted = 255 - gray

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    detected = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)

    cleaned = cv2.subtract(inverted, detected)

    return 255 - cleaned


def run_pipeline(source, save_steps=False):
    steps = {}

    img = load_image(source)
    if save_steps: steps["1_loaded"] = img.copy()

    img = to_grayscale(img)
    if save_steps: steps["2_grayscale"] = img.copy()

    img = remove_horizontal_lines(img)
    if save_steps: steps["2b_lines_removed"] = img.copy()

    img = enhance_contrast(img, 1.6)
    if save_steps: steps["3_contrast"] = img.copy()

    img = remove_noise(img, "nlmeans", 12)
    if save_steps: steps["4_denoised"] = img.copy()

    img = apply_threshold(img, "adaptive")
    if save_steps: steps["5_threshold"] = img.copy()

    img, angle = deskew(img)
    if save_steps: steps["6_deskewed"] = img.copy()

    img = resize_for_ocr(img, 1.8)
    if save_steps: steps["7_resized"] = img.copy()

    img = add_border(img, 20)
    if save_steps: steps["8_final"] = img.copy()

    return img, angle, steps