import cv2
import math
import numpy as np
import os


def convert0(number): #A
    if number == "0":
        return "take note"
    elif number == "1":
        return "use the laptop"
    elif number == "2":
        return "use the cellphone"
    elif number == "6":
        return "raise a hand"
    elif number == "7":
        return "tilt head"
    elif number == "8":
        return "support head"
    elif number == "9":
        return "sleep"
    elif number == "10":
        return "use the laptop"


def convert1(number): #D
    if number == "0":
        return "take note"
    elif number == "1":
        return "use the laptop"
    elif number == "2":
        return "take note"
    elif number == "6":
        return "raise a hand"
    elif number == "7":
        return "tilt head"
    elif number == "8":
        return "support head"
    elif number == "9":
        return "sleep"
    elif number == "10":
        return "talk on the phone"


def convert(number): #D
    if number == "0":
        return "take note"
    elif number == "1":
        return "use the laptop"
    elif number == "2":
        return "use the cellphone"
    elif number == "6":
        return "raise a hand"
    elif number == "7":
        return "tilt head"
    elif number == "8":
        return "support head"
    elif number == "9":
        return "sleep"
    elif number == "10":
        return "talk on the phone"


def most_common(lst):
    return max(set(lst), key=lst.count)

cap = cv2.VideoCapture("00031.MTS")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('action1.avi', fourcc, 30.0, (700, 700))
success = True

read_file0 = open("./action/0_v4.txt", "r")
lines0 = read_file0.read().splitlines()
read_file1 = open("./action/1_v4.txt", "r")
lines1 = read_file1.read().splitlines()
read_file2 = open("./action/2_v4.txt", "r")
lines2 = read_file2.read().splitlines()
read_file3 = open("./action/3_v4.txt", "r")
lines3 = read_file3.read().splitlines()

count = 0
while success:
    success, image = cap.read()
    if success:
        img = np.zeros((700, 700, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if count >= 30:
            label0 = most_common(lines0[count - 30:count + 1])
            label1 = most_common(lines1[count - 30:count + 1])
            label2 = most_common(lines2[count - 30:count + 1])
            label3 = most_common(lines3[count - 30:count + 1])
        else:
            label0 = lines0[count]
            label1 = lines1[count]
            label2 = lines2[count]
            label3 = lines3[count]
        cv2.putText(img, "student A", (0, 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, convert0(label0), (0, 125), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "student B", (0, 200), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, convert(label3), (0, 275), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "student C", (0, 350), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, convert(label2), (0, 425), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "student D", (0, 500), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, convert1(label1), (0, 575), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(img)
        count += 1

cap.release()
out.release()
