import cv2
import pytesseract
from PIL import Image

img = cv2.imread("test1.png")

# increase size
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

cv2.imwrite("big.png", img)
