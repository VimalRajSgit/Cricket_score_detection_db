import cv2 as cv
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

cap = cv.VideoCapture('cook_batting.mp4')

fps = cap.get(cv.CAP_PROP_FPS)          # frames per second
interval = int(fps * 3)                 # 3 seconds
frame_count = 0

cv.namedWindow('Resized Video', cv.WINDOW_NORMAL)
cv.resizeWindow('Resized Video', 800, 600)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    start_x = int(w * 0.05)
    start_y = int(h * 0.90)
    end_x = int(w * 0.95)
    end_y = h

    cv.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 2)

    # OCR every 3 seconds
    if frame_count % interval == 0:

        roi = frame[start_y:end_y, start_x:end_x]

        # resize for better OCR
        roi = cv.resize(roi, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

        img = Image.fromarray(roi)

        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.-()*'

        text = pytesseract.image_to_string(img, config=config)

        print("Detected:", text.strip())

    frame_count += 1

    cv.imshow('Resized Video', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
