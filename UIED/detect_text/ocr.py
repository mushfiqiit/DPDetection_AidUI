import cv2
import os
import requests
import json
from base64 import b64encode
import time
import pytesseract
from PIL import Image


def Google_OCR_makeImageData(imgpath):
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                # 'type': 'TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()


def ocr_detection_google(imgpath):
    # Start the timer
    start = time.time()
    
    # Load the image using PIL
    try:
        img = Image.open(imgpath)
    except FileNotFoundError:
        print(f"File not found: {imgpath}")
        return None
    
    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(img)
    
    # Print the time taken
    print(f"*** Text Detection Time Taken: {time.time() - start:.3f}s ***")
    
    # Split the text into lines (or words) for individual text annotations
    lines = text.split('\n')
    
    # Create a JSON response similar to Google OCR
    text_annotations = [{'description': line} for line in lines if line.strip()]
    response = {
        'responses': [{
            'textAnnotations': text_annotations
        }]
    }
    
    # Return the relevant part of the response
    return response['responses'][0]['textAnnotations']

""" # Example usage
imgpath = 'testing.png'
json_response = ocr_detection_google(imgpath)
print(json.dumps(json_response, indent=2)) """

"""     start = time.clock()
    url = 'https://vision.googleapis.com/v1/images:annotate'
    # api_key = 'AIzaSyDUc4iOUASJQYkVwSomIArTKhE2C6bHK8U'             # *** Replace with your own Key ***
    api_key = 'AIzaSyAeSaaOE-upsRshfOEkkMIUcAiBzDSVOAo'
    imgdata = Google_OCR_makeImageData(imgpath)
    response = requests.post(url,
                             data=imgdata,
                             params={'key': api_key},
                             headers={'Content_Type': 'application/json'})
    # print('*** Text Detection Time Taken:%.3fs ***' % (time.clock() - start))
    print("*** Please replace the Google OCR key at detect_text/ocr.py line 28 with your own (apply in https://cloud.google.com/vision) ***")
    if response.json()['responses'] == [{}]:
        # No Text
        return None
    else:
        return response.json()['responses'][0]['textAnnotations'][1:]
 """