import collections
import cv2
import numpy as np
import easyocr
from json import JSONEncoder

class OCRscan:
    def __init__(self):
        self.reader = easyocr.Reader(['en'],gpu=True)  # Initialize easyocr reader with English language

    def scan(self, img):
        results = self.reader.readtext(img, detail=1)  # Using easyocr to read text from image
        out = collections.defaultdict(list)
        text_list = []
        
        for (bbox, text, prob) in results:
            (tl, tr, br, bl) = bbox
            x = int(tl[0])
            y = int(tl[1])
            w = int(br[0] - tl[0])
            h = int(br[1] - tl[1])
            text = str(text).strip()
            text_list.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': text})
            out["left"].append(x)
            out["top"].append(y)
            out["width"].append(w)
            out["height"].append(h)
            out["text"].append(text)
            # Draw the bounding box on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Optionally, you can draw the text on the image as well
            # cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        self.displayImg(img)
        return img, out, text_list

    def displayImg(self, img):
        img = self.resizeImg(img)
        cv2.imshow('ocr', img)
        cv2.waitKey(0)

    def resizeImg(self, img):
        scale_percent = 50  # resize to 50% of the original size
        window_width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dsize = (window_width, height)
        img = cv2.resize(img, dsize)
        return img

    # The following methods can be used for image preprocessing if needed.
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    def thresholding(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def dilate(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    def erode(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    def opening(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def canny(self, image):
        return cv2.Canny(image, 100, 200)

