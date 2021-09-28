# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import tensorflow as tf
import cifar10
import cifar10_input
import time
import csv
from sys import platform
from PIL import Image
from skimage import io
from skimage.transform import resize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Remember to add your installation path here
# Option a
dir_path = os.path.dirname(os.path.realpath(__file__))
#if platform == "win32": sys.path.append(dir_path + '/../../python/openpose/');
#else: sys.path.append('../../python');
sys.path.append(dir_path + '/openpose/build_windows/python/openpose/')
# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
# sys.path.append('/usr/local/python')

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "COCO"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = dir_path + "/../../../models/"
# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)


name = dict([["Nose", 0], ["Neck", 1], ["RShoulder", 2], ["RElbow", 3], ["RWrist", 4], ["LShoulder", 5], ["LElbow", 6],
            ["LWrist", 7], ["RHip", 8], ["RKnee", 9], ["RAnkle", 10], ["LHip", 11], ["LKnee", 12], ["LAnkle", 13],
            ["REye", 14], ["LEye", 15], ["REar", 16], ["LEar", 17], ["Bkg", 18]])
arm_list_name = ["LWrist", "LElbow", "LShoulder", "Neck", "RShoulder", "RElbow", "RWrist"]
leg_list_name = ["LHip", "Neck", "RHip"]
head_list_name = ["LEar", "LEye", "Nose", "REye", "REar"]


def skeleton_norm(data):
    shoulder_width = la.norm(data[name["LShoulder"]] - data[name["RShoulder"]])
    origin = data[name["Neck"]][0], data[name["Neck"]][1]
    norm_data = np.copy(data)
    for i in range(len(data)):
        if shoulder_width == 0:
            continue
        norm_data[i][0] = (data[i][0] - origin[0]) / (shoulder_width * 1.2 / 50)
        norm_data[i][1] = (data[i][1] - origin[1]) / (shoulder_width * 1.2 / 50)
    return norm_data


def draw_skeleton(data, filename):
    fig = plt.figure(2)
    fig.set_size_inches(5, 5)
    fig.canvas.set_window_title("frame number :2")

    arm_list = []
    for each_joint in arm_list_name:
        if data[name[each_joint]][2] != 0:
            arm_list.append(data[name[each_joint]])

    leg_list = []
    for each_joint in leg_list_name:
        if data[name[each_joint]][2] != 0:
            leg_list.append(data[name[each_joint]])

    head_list = []
    for each_joint in head_list_name:
        if data[name[each_joint]][2] != 0:
            head_list.append(data[name[each_joint]])

    arm_list = np.array(arm_list)
    leg_list = np.array(leg_list)
    head_list = np.array(head_list)

    plt.plot(arm_list[:, 0], arm_list[:, 1], color='g')
    plt.plot(leg_list[:, 0], leg_list[:, 1], color='b')
    plt.plot(head_list[:, 0], head_list[:, 1], color='r')
    tmp_point_list = np.append([data[name['Neck']]], [data[name['Nose']]], axis=0)
    plt.plot(tmp_point_list[:, 0], tmp_point_list[:, 1], color='r')

    plt.figure(2, figsize=(1, 1))

    plt.autoscale(False)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(filename, pad_inches=0)
    # fig.canvas.draw()
    # skeleton_img = np.array(fig.canvas.renderer._renderer)
    plt.clf()


def evaluate(im_path):
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, shape=(1, 32, 32, 3), name='images')
        labels = tf.placeholder(tf.int32, shape=1, name='labels')

        logits = cifar10.inference(images)
        pred_lab = tf.argmax(logits, 1)

        variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        with tf.Session(config=config) as sess:

            ckpt = tf.train.get_checkpoint_state("./model/pose/")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
            im = Image.open(im_path)
            im = im.resize((32, 32), Image.BILINEAR)
            im = (np.array(im))
            # io.imsave("./test2.jpg", im)
            # im = cv2.resize(im, (32, 32))
            batch_x = tf.image.per_image_standardization(im).eval()

            batch_x = [batch_x]
            batch_y = [1]

            img_label = pred_lab.eval(feed_dict={images: batch_x, labels: batch_y})
            sess.close()

        return img_label


def crop_object(im, data, flag):
    shoulder_width = la.norm(data[name["LShoulder"]] - data[name["RShoulder"]])
    if flag:
        range = int(shoulder_width)
        if data[name["RWrist"]][0] == 0:
            wrist_mid = int(data[name["LWrist"]][0] - ((data[name["LShoulder"]][0] - data[name["RShoulder"]][0]) / 2)), int(data[name["LWrist"]][1])
        elif data[name["LWrist"]][0] == 0:
            wrist_mid = int(data[name["RWrist"]][0] + ((data[name["LShoulder"]][0] - data[name["RShoulder"]][0]) / 2)), int(data[name["RWrist"]][1])
        else:
            wrist_mid = int((data[name["RWrist"]][0] + data[name["LWrist"]][0]) / 2), int((data[name["RWrist"]][1] + data[name["LWrist"]][1]) / 2)
        y1 = 0 if wrist_mid[1] - range < 0 else wrist_mid[1] - range
        y2 = 1080 if wrist_mid[1] + range > 1080 else wrist_mid[1] + range
        x1 = 0 if wrist_mid[0] - range < 0 else wrist_mid[0] - range
        x2 = 1920 if wrist_mid[0] + range > 1920 else wrist_mid[0] + range
    else:
        range = int(shoulder_width / 2)
        if data[name["RWrist"]][1] < data[name["LWrist"]][1]:
            if data[name["RWrist"]][1] is 0:
                y1 = 0 if int(data[name["LWrist"]][1]) - range * 2 < 0 else int(data[name["LWrist"]][1]) - range * 2
                y2 = int(data[name["LWrist"]][1])
                x1 = 0 if int(data[name["LWrist"]][0]) - range < 0 else int(data[name["LWrist"]][0]) - range
                x2 = 1920 if int(data[name["LWrist"]][0]) + range > 1920 else int(data[name["LWrist"]][0]) + range
            else:
                y1 = 0 if int(data[name["RWrist"]][1]) - range * 2 < 0 else int(data[name["RWrist"]][1]) - range * 2
                y2 = int(data[name["RWrist"]][1])
                x1 = 0 if int(data[name["RWrist"]][0]) - range < 0 else int(data[name["RWrist"]][0]) - range
                x2 = 1920 if int(data[name["RWrist"]][0]) + range > 1920 else int(data[name["RWrist"]][0]) + range
        else:
            if data[name["RWrist"]][1] is 0:
                y1 = 0 if int(data[name["RWrist"]][1]) - range * 2 < 0 else int(data[name["RWrist"]][1]) - range * 2
                y2 = int(data[name["RWrist"]][1])
                x1 = 0 if int(data[name["RWrist"]][0]) - range < 0 else int(data[name["RWrist"]][0]) - range
                x2 = 1920 if int(data[name["RWrist"]][0]) + range > 1920 else int(data[name["RWrist"]][0]) + range
            else:
                y1 = 0 if int(data[name["LWrist"]][1]) - range * 2 < 0 else int(data[name["LWrist"]][1]) - range * 2
                y2 = int(data[name["LWrist"]][1])
                x1 = 0 if int(data[name["LWrist"]][0]) - range < 0 else int(data[name["LWrist"]][0]) - range
                x2 = 1920 if int(data[name["LWrist"]][0]) + range > 1920 else int(data[name["LWrist"]][0]) + range
    return im[y1:y2, x1:x2]


def object_classification(im, model_name):
    if im.shape[0] == 0 or im.shape[1] == 0:
        return 4
    save_dir = './keras_model/'
    model_path = os.path.join(save_dir, model_name)
    model = load_model(model_path)
    im = im / 255.0
    im = resize(im, (224, 224))
    x = im.reshape((1, 224, 224, 3))
    result = np.argmax(model.predict(x), axis=1)
    del model
    K.clear_session()
    return result[0]


def write_log(txt_filename, content):
    text_file = open(txt_filename, "a")
    text_file.write(content + "\n")
    text_file.close()

ref_points = list()
ref_points.append([327.17783, 477.22635])
ref_points.append([1589.9675, 436.1017])
ref_points.append([945.23645, 456.64346])
ref_points.append([494.95197, 362.4041])
cap = cv2.VideoCapture("./00031.MTS")
success = True
count = 0
while success:
    success, img = cap.read()
    if success:
        if count > 1475:
            # Read new image
            # img = cv2.imread("./cellphone_000000000000_rendered.png")
            # img1 = io.imread("./cellphone_000000000000_rendered.png")
            np.set_printoptions(suppress=True)
            # Output keypoints and the image with the human skeleton blended on it
            keypoints, output_image = openpose.forward(img, True)
            # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image)
            bone_index = 0
            for each in keypoints:
                person_index = bone_index
                if count == 0:
                    ref_points.append(each[name["Neck"]])
                else:
                    for j in range(len(ref_points)):
                        if la.norm(each[name["Neck"]][0:2] - ref_points[j]) < 50:
                            ref_points[j] = each[name["Neck"]][0:2]
                            person_index = j
                            break
                if person_index > 3:
                    break
                content = list()
                content.append(count)
                for n in each:
                    content.append(n[0])
                    content.append(n[1])
                    content.append(n[2])
                with open("./keypoint/" + str(person_index) + "/Joints.csv", "a", newline='') as file:
                    csv_file = csv.writer(file)
                    csv_file.writerow(content)
                    file.close()
                norm_each = skeleton_norm(each)

                draw_skeleton(norm_each, "./bone_image/" + str(person_index) + "/" + str(count) + ".jpg")
                label = evaluate("./bone_image/" + str(person_index) + "/" + str(count) + ".jpg")
                time.sleep(1)
                if label == 0:
                    object_img = crop_object(img, each, True)
                    filename = os.path.join("E:/Desktop/system/crop1/", str(person_index)) + "/" + str(count) + ".jpg"
                    cv2.imwrite(filename, object_img)
                    write_log(str(person_index) + ".txt", "0")
                elif label == 1:
                    write_log(str(person_index) + ".txt", "6")
                elif label == 2:
                    write_log(str(person_index) + ".txt", "7")
                elif label == 3:
                    object_img = crop_object(img, each, False)
                    filename = os.path.join("E:/Desktop/system/crop2/", str(person_index)) + "/" + str(count) + ".jpg"
                    cv2.imwrite(filename, object_img)
                    write_log(str(person_index) + ".txt", "8")
                elif label == 4:
                    write_log(str(person_index) + ".txt", "9")
                bone_index += 1
                print(count)
            count += 1
        else:
            count += 1


