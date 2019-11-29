import cv2
import numpy as np

def clahe(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
    img2 = clahe.apply(img)
    res = np.hstack((img, img2))
    return res # Grayscale

def postclahe(img):
    median_clahe = cv2.medianBlur(img, 5)
    ret, th1 = cv2.threshold(median_clahe, 127, 255, cv2.THRESH_BINARY)
    return th1 # Grayscale