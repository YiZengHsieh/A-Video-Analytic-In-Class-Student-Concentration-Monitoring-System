import cv2
import math
import numpy as np
import os


def most_common(lst):
    return max(set(lst), key=lst.count)

cap = cv2.VideoCapture("00031.MTS")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('yawn.avi', fourcc, 30.0, (700, 700))
success = True

read_file0 = open("./1932/yawn/0_yawn.txt", "r")
lines0 = read_file0.read().splitlines()
read_file1 = open("./1932/yawn/1_yawn.txt", "r")
lines1 = read_file1.read().splitlines()
read_file2 = open("./1932/yawn/2_yawn.txt", "r")
lines2 = read_file2.read().splitlines()
read_file3 = open("./1932/yawn/3_yawn.txt", "r")
lines3 = read_file3.read().splitlines()
yawn0 = 0
yawn1 = 0
yawn2 = 0
yawn3 = 0

count = 0
while success:
    success, image = cap.read()
    if success:
        img = np.zeros((700, 700, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if float(lines0[count].split(",")[1]) >= 40:
            yawn0 += 1
            if yawn0 >= 45:
                print("yawn")
                cv2.putText(img, "yawning", (0, 100), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        elif float(lines0[count].split(",")[1]) < 40:
            yawn0 = 0

        if float(lines1[count].split(",")[1]) >= 40:
            yawn1 += 1
            if yawn1 >= 45:
                print("yawn")
                cv2.putText(img, "yawning", (0, 400), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        elif float(lines1[count].split(",")[1]) < 40:
            yawn1 = 0

        if float(lines2[count].split(",")[1]) >= 40:
            yawn2 += 1
            if yawn1 >= 45:
                print("yawn")
                cv2.putText(img, "yawning", (0, 250), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        elif float(lines2[count].split(",")[1]) < 40:
            yawn2 = 0

        if float(lines3[count].split(",")[1]) >= 40:
            yawn3 += 1
            if yawn3 >= 45:
                print("yawn")
                cv2.putText(img, "yawning", (0, 550), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        elif float(lines3[count].split(",")[1]) < 40:
            yawn3 = 0

        out.write(img)
        count += 1

cap.release()
out.release()
