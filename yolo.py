# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import subprocess
import time
from sys import platform
from PIL import Image
from skimage import io
from skimage.transform import resize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def write_log(txt_filename, content):
    text_file = open(txt_filename, "a")
    text_file.write(content + "\n")
    text_file.close()


person_index = "3"
read_file = open("./" + person_index + "_v3.txt", "r")
lines = read_file.read().splitlines()
count = 0
prev = "8"
print(len(lines))
for each in lines:
    if each == "8":
        if os.path.exists("./crop2/" + person_index + "/" + str(count) + ".jpg"):
            filename = "E:/Desktop/system/crop2/" + person_index + "/" + str(count) + ".jpg"
            os.chdir("../darknet/build/darknet/x64/")
            # yolo = os.popen("powershell.exe ./darknet.exe detector test data/obj.data yolo-obj-test-newanchor.cfg backup/yolo-obj-newanchor_23972_0.917123.weights -dont_show -thresh 0.5 " + filename)
            yolo = os.popen("powershell.exe ./darknet.exe detector test data/obj2.data yolo-obj2-test.cfg backup/yolo-obj2_29169_0.173446.weights -dont_show -thresh 0.4 " + filename)
            output = yolo.read()
            os.chdir("E:/Desktop/system/")
            if len(output.split("\n")) == 4:
                '''
                if output.split("\n")[2].split(":")[0] == "cellphone":
                    prev = "2"
                    write_log(person_index + "_v3.txt", "2")
                elif output.split("\n")[2].split(":")[0] == "laptop":
                    prev = "1"
                    write_log(person_index + "_v3.txt", "1")
                else:
                    prev = "0"
                    write_log(person_index + "_v3.txt", "0")
                '''
                if output.split("\n")[2].split(":")[0] == "cellphone":
                    prev = "10"
                    write_log(person_index + "_v4.txt", "10")
                else:
                    prev = "8"
                    write_log(person_index + "_v4.txt", "8")
            elif len(output.split("\n")) > 4:
                max_score = 0
                index = 2
                for i in range(2, len(output.split("\n")) - 1):
                    print(output.split("\n")[i].split(":"))
                    if int(output.split("\n")[i].split(":")[1]) > max_score:
                        index = i
                '''
                if output.split("\n")[i].split(":")[0] == "cellphone":
                    prev = "2"
                    write_log(person_index + "_v3.txt", "2")
                elif output.split("\n")[i].split(":")[0] == "laptop":
                    prev = "1"
                    write_log(person_index + "_v3.txt", "1")
                else:
                    prev = "0"
                    write_log(person_index + "_v3.txt", "0")
                '''
                if output.split("\n")[2].split(":")[0] == "cellphone":
                    prev = "10"
                    write_log(person_index + "_v4.txt", "10")
                else:
                    prev = "8"
                    write_log(person_index + "_v4.txt", "8")
            else:
                prev = "8"
                write_log(person_index + "_v4.txt", "8")
            yolo.close()
            '''
            if label[0] == 0:
                prev = "10"
                write_log(person_index + "_v2.txt", "10")
            elif label[0] == 1:
                prev = "8"
                write_log(person_index + "_v2.txt", "8")
            '''
        else:
            write_log(person_index + "_v4.txt", prev)
    else:
        write_log(person_index + "_v4.txt", each)
    count += 1
read_file.close()
'''
filename = os.path.join("E:/Desktop/system/crop/", str(person_index)) + "_" + str(count) + ".jpg"
cv2.imwrite(filename, object_img)
os.chdir("../darknet/build/darknet/x64/")
output = os.popen("powershell.exe ./darknet.exe detector test data/obj2.data yolo-obj2-test.cfg backup/yolo-obj2_29169_0.173446.weights -dont_show -thresh 0.4 " + filename).read()
if len(output.split("\n")) > 2:
    if output.split("\n")[2].split(":")[0] == "cellphone":
        write_log(str(person_index) + ".txt", "10")
    else:
        write_log(str(person_index) + ".txt", "8")
else:
    write_log(str(person_index) + ".txt", "8")
'''


