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
cap = cv2.VideoCapture("./00031.MTS")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('headpose.avi', fourcc, 30.0, (550, 550))
success = True
count = 0
yawn_landmark = list()
read_file = open("./head_pose/1_angle_video.txt", "r")
lines = read_file.read().splitlines()
# print(lines[0].split(",")[1])
yawn_count = 0
font = cv2.FONT_HERSHEY_SIMPLEX
yaw_list = []
pitch_list = []
while success:
    success, imgs = cap.read()
    if success:
        img = np.zeros((550, 550, 3), np.uint8)
        cv2.rectangle(img, (175, 25), (375, 225), (255, 255, 255), 2)
        cv2.rectangle(img, (25, 425), (525, 525), (255, 255, 255), 2)
        if count > 2950:
            break
        if count > 2749:
            yaw = float(lines[count - 2750].split(",")[1])
            pitch = float(lines[count - 2750].split(",")[2])
            if pitch < -15:
                cv2.circle(img, (int(275 - 10 * math.cos((yaw / 360) * math.pi)), 125 + 10), 20, (0, 0, 255), -1)
            else:
                print(250 * math.cos((yaw / 360) * math.pi), yaw)
                cv2.circle(img, (int(275 - 150 * math.cos((yaw / 360) * math.pi)), 475), 20, (0, 0, 255), -1)
            out.write(img)
            # norm_face = normalization(face)
    count += 1
cap.release()
out.release()
