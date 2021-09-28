import cv2
import math
import numpy as np
import os


def most_common(lst):
    return max(set(lst), key=lst.count)

cap = cv2.VideoCapture("00031.MTS")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test.avi', fourcc, 30.0, (1920, 1920))
success = True
count = 0
while success:
    success, image = cap.read()
    if success:
        img = np.zeros((1920, 1920, 3), np.uint8)
        img[0:1080, 0:1920] = image
        out.write(img)
        count += 1

cap.release()
out.release()
