import cv2 as cv
import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def preprocess_roi(roi):
    """Handle mixed dark/light scoreboard zones"""
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # CLAHE for local contrast (fixes both dark and light zones)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv.filter2D(gray, -1, kernel)

    # Denoise
    gray = cv.fastNlMeansDenoising(gray, h=10)

    # Scale up 3x for better OCR
    gray = cv.resize(gray, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)

    h, w = gray.shape
    results = []

    # Split into zones (left ~15%, center ~70%, right ~15% are approx dark/light/dark)
    # Better: detect by average brightness
    zone_w = w // 3  # rough split into 3 zones
    zones = [(0, zone_w), (zone_w, 2 * zone_w), (2 * zone_w, w)]

    for x1, x2 in zones:
        zone = gray[:, x1:x2]
        avg = np.mean(zone)

        if avg < 127:
            # Dark background → invert before threshold
            zone = cv.bitwise_not(zone)

        _, binary = cv.threshold(zone, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Small morphological cleanup
        kernel_m = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel_m)

        results.append(binary)

    # Stitch zones back together
    combined = np.hstack(results)
    return combined

cap = cv.VideoCapture('stokes.mp4')
fps = cap.get(cv.CAP_PROP_FPS)
interval = int(fps * 3)
frame_count = 0

config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.-()*: '

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    start_x, start_y = int(w * 0.05), int(h * 0.90)
    end_x, end_y = int(w * 0.95), h

    if frame_count % interval == 0:
        roi = frame[start_y:end_y, start_x:end_x]
        processed = preprocess_roi(roi)

        img = Image.fromarray(processed)
        text = pytesseract.image_to_string(img, config=config)
        print(f"[{frame_count}] {text.strip()}")

    frame_count += 1

cap.release()
