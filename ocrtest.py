
"""
Created on Wed Jun 12 22:20:16 2019

@author: rahul
"""
import cv2
import pytesseract
import argparse
#from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Enter path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
print(image) #to check for correct path or image is loaded or not
text=pytesseract.image_to_string(image,lang='eng')
print(text)

#USAGE- MAIN------ on terminal write--- python ocrtest.py --image ../example.png