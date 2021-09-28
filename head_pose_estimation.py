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

# Network Parameters
n_hidden_1 = 5  # 1st layer number of neurons
n_hidden_2 = 5  # 2nd layer number of neurons
n_input = 8  # MNIST data input (img shape: 28*28)
n_classes = 1  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

init = tf.global_variables_initializer()


def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


def angle1(temp):
    return (temp * 60) - 30


def angle2(temp):
    return (temp * 90) - 45


def eval_angle(feature):
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    with tf.Session(config=config) as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state("./model/yaw/")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        logits = multilayer_perceptron(X)
        result = logits.eval(feed_dict={X: feature, keep_prob: 1.0})
        pred_yaw = angle2(result[0])
        sess.close()

    with tf.Session(config=config) as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state("./model/pitch/")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        logits = multilayer_perceptron(X)
        result = logits.eval(feed_dict={X: feature, keep_prob: 1.0})
        pred_pitch = angle1(result[0])
        sess.close()
    return pred_yaw, pred_pitch


def normalization(landmark):
    landmark = np.array(landmark).astype(int)
    offset = (landmark[36][0] + landmark[45][0]) / 2, (landmark[36][1] + landmark[45][1]) / 2
    unit = math.sqrt((landmark[36][0] - landmark[45][0]) ** 2 + (landmark[36][1] - landmark[45][1]) ** 2)
    for each in landmark:
        each[0] = (each[0] - offset[0]) * (average_unit / unit)
        each[1] = (each[1] - offset[1]) * (average_unit / unit)
    return landmark


def head_pose(landmark):
    landmark_list = [landmark[30], landmark[36], landmark[45], landmark[48], landmark[54]]
    landmark_list = np.array(landmark_list).astype(int)
    feature = []
    for each in landmark_list[1:]:
        distance = math.sqrt((each[0] - landmark_list[0][0]) ** 2 + (each[1] - landmark_list[0][1]) ** 2)
        angle = math.acos(abs(each[0] - landmark_list[0][0]) / distance)
        feature.append(distance)
        feature.append(angle / math.pi)

    feature = np.array(feature)
    feature = np.reshape(feature, (1, 8))

    return feature


def calc_EAR(eye):
    vertical = math.sqrt((eye[5][0] - eye[1][0]) ** 2 + (eye[5][1] - eye[1][1]) ** 2) + math.sqrt((eye[4][0] - eye[2][0]) ** 2 + (eye[4][1] - eye[2][1]) ** 2)
    horizon = math.sqrt((eye[3][0] - eye[0][0]) ** 2 + (eye[3][1] - eye[0][1]) ** 2)
    return vertical / (2 * horizon)


def calc_mouth_h(up_lip, bottom_lip):
    distance = math.sqrt((up_lip[0] - bottom_lip[0]) ** 2 + (up_lip[1] - bottom_lip[1]) ** 2)
    return distance


def write_log(txt_filename, content):
    text_file = open(txt_filename, "a")
    text_file.write(content + "\n")
    text_file.close()

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False, use_cnn_face_detector=True)
ref_points = list()
ref_points.append([291, 390])
ref_points.append([978, 404])
ref_points.append([492, 270])
ref_points.append([1573, 367])
cap = cv2.VideoCapture("./00031.MTS")
success = True
count = 0
while success:
    success, img = cap.read()
    if success:
        if count > 3562:
            # f_dir = "./personne01146+0+0.jpg"
            img = img[..., ::-1]
            preds = fa.get_landmarks(img, all_faces=True)
            # pred = fa.detect_faces(img)

            # img1 = cv2.imread(f_dir)
            face_index = 0
            flag = [0, 0, 0, 0]
            for face in preds:
                person_index = -1
                if count == 0:
                    ref_points.append(face[0])
                else:
                    distance = 120
                    for j in range(len(ref_points)):
                        if la.norm(face[0] - ref_points[j]) < distance and flag[j] == 0:
                            ref_points[j] = face[0]
                            person_index = j
                            distance = la.norm(face[0] - ref_points[j])
                if person_index < 0:
                    break
                flag[person_index] = 1
                filename = "./face/" + str(person_index) + "/" + str(count) + ".jpg"
                x1 = int(face[0][0])
                x2 = int(face[0][0])
                y1 = int(face[0][1])
                y2 = int(face[0][1])
                for each in face:
                    x1 = int(each[0]) if each[0] < x1 else x1
                    x2 = int(each[0]) if each[0] > x2 else x2
                    y1 = int(each[1]) if each[1] < y1 else y1
                    y2 = int(each[1]) if each[1] > y2 else y2
                io.imsave(filename, img[y1:y2, x1:x2])
                content = list()
                content.append(count)
                for n in face:
                    content.append(n[0])
                    content.append(n[1])
                with open("./landmark/" + str(person_index) + "_landmark.csv", "a", newline='') as file:
                    csv_file = csv.writer(file)
                    csv_file.writerow(content)
                    file.close()
                norm_face = normalization(face)
                face_feature = head_pose(norm_face)
                yaw, pitch = eval_angle(face_feature)
                write_log("./head_pose/" + str(person_index) + "_angle.txt", str(count) + "," + str(yaw) + "," + str(pitch))
                left_eye_EAR = calc_EAR(norm_face[36:42])
                right_eye_EAR = calc_EAR(norm_face[42:48])
                write_log("./EAR/" + str(person_index) + "_EAR.txt", str(count) + "," + str(left_eye_EAR) + "," + str(right_eye_EAR))
                mouth_height = calc_mouth_h(norm_face[51], norm_face[57])
                write_log("./yawn/" + str(person_index) + "_yawn.txt", str(count) + "," + str(mouth_height))
                face_index += 1
                print(count)
                print(ref_points[person_index])
        count += 1
