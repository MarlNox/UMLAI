import cv2
import numpy as np
import easyocr

# Initialize easyocr reader with English language
reader = easyocr.Reader(['en'])

img = cv2.imread('test2_big.png')

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)

# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# OCR with easyocr
results = reader.readtext(img, detail=1)  # Using easyocr to read text from image
for (bbox, text, prob) in results:
    (tl, tr, br, bl) = bbox
    x = int(tl[0])
    y = int(tl[1])
    w = int(br[0] - tl[0])
    h = int(br[1] - tl[1])
    text = str(text).strip()
    print("Text: {}, Prob: {:.2f}, Top Left: {}, {}, Bottom Right: {}, {}".format(text, prob, x, y, x + w, y + h))
    # Draw the bounding box on the image
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Optionally, you can draw the text on the image as well
    # cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

# Resize and display the image
scale_percent = 50  # resize to 50% of the original size
window_width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dsize = (window_width, height)
resized_img = cv2.resize(img, dsize)
cv2.imshow('img', resized_img)
cv2.waitKey(0)
