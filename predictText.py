import pytesseract
import cv2
import os
import re


def getlicense():
    arr = []
    for i in os.listdir('output'):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        image = cv2.imread('output/'+i)
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(image, config=config)
        arr.append(text)

    li = []
    for x in arr:
        txt = ""
        z = 0
        for i in x:
            if i.isdigit():
                z = 1
                txt += i
        if z == 1:
            z = 0
            li.append(txt)

    frequency = {}
    for item in li:
        if item in frequency:          
            frequency[item] += 1
        else:
            frequency[item] = 1

    max1 = max(frequency, key=frequency.get)
    print(max1)
    return max1