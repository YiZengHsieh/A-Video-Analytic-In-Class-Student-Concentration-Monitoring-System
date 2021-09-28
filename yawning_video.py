#!/usr/bin/env python

import cv2
import math
import numpy as np
import numpy.linalg as la
import os
import face_alignment
import csv
import tensorflow as tf
from skimage import io, draw
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

average_unit = 69.9522282268

def normalization(landmark):
    landmark = np.array(landmark).astype(int)
    offset = (landmark[36][0] + landmark[45][0]) / 2, (landmark[36][1] + landmark[45][1]) / 2
    unit = math.sqrt((landmark[36][0] - landmark[45][0]) ** 2 + (landmark[36][1] - landmark[45][1]) ** 2)
    for each in landmark:
        each[0] = (each[0] - offset[0]) * (average_unit / unit)
        each[1] = (each[1] - offset[1]) * (average_unit / unit)
    return landmark


def calc_mouth_h(up_lip, bottom_lip):
    distance = math.sqrt((up_lip[0] - bottom_lip[0]) ** 2 + (up_lip[1] - bottom_lip[1]) ** 2)
    return distance


def write_log(txt_filename, content):
    text_file = open(txt_filename, "a")
    text_file.write(content + "\n")
    text_file.close()

# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False, use_cnn_face_detector=True)
ref_points = 881, 330
cap = cv2.VideoCapture("./00031.MTS")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('yawn.avi', fourcc, 30.0, (1920, 1080))
success = True
count = 0
yawn_landmark = list()
with open('landmark/1_yawn.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        yawn_landmark.append(row)
read_file = open("./yawn/1_yawn_video.txt", "r")
lines = read_file.read().splitlines()
# print(lines[0].split(",")[1])
yawn_count = 0
font = cv2.FONT_HERSHEY_SIMPLEX
while success:
    success, img = cap.read()
    if success:
        if count > 1699:
            break
        if count > 1399:
            x1 = int(yawn_landmark[count - 1400][97]) - 10
            x2 = int(yawn_landmark[count - 1400][109]) + 10
            y1 = int(yawn_landmark[count - 1400][104]) - 10
            y2 = int(yawn_landmark[count - 1400][116]) + 10
            if float(lines[count - 1400].split(",")[1]) > 24:
                yawn_count += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                if yawn_count >= 45:
                    cv2.putText(img, 'yawning!', (x2 + 10, y1), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                yawn_count = 0
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            out.write(img)
            # norm_face = normalization(face)
    count += 1
cap.release()
out.release()
