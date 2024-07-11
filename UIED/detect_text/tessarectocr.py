import cv2
import pytesseract
from text_detection import text_detection


def ocr_detection_tesseract(imgpath):
    # Load image using OpenCV
    img = cv2.imread(imgpath)

    # Convert image to RGB (OpenCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(img)

    # You can also get bounding boxes and other details
    # d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    return text

# Example usage
""" imgpath = '/home/mushfiqur/DPDetection_AidUI/UIED/detect_text/testing.png'
text = ocr_detection_tesseract(imgpath)
print(text) """

