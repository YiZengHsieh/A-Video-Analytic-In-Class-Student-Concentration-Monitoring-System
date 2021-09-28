'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

import keras
import os
import numpy as np
import cv2
import math
import csv
from keras.optimizers import SGD
from keras import applications
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, activations
from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from vis.visualization import visualize_cam
from vis.utils import utils
import matplotlib.pyplot as plt
import time
from skimage import io
from skimage.transform import resize


def write_log(txt_filename, content):
    text_file = open(txt_filename, "a")
    text_file.write(content + "\n")
    text_file.close()

# dimensions of our images.
img_width, img_height = 224, 224
person_index = "3"
save_dir = './keras_model/'
model_name = '2class_trainall.h5'
# dir_name = "./crop1/0"
# dirs = os.walk(dir_name).__next__()[2]
model_path = os.path.join(save_dir, model_name)
model = load_model(model_path)

read_file = open("./" + person_index + "_v1.txt", "r")
lines = read_file.read().splitlines()
count = 0
prev = "8"
for each in lines:
    if each == "8":
        if os.path.exists("./crop2/" + person_index + "/" + str(count) + ".jpg"):
            img = io.imread("./crop2/" + person_index + "/" + str(count) + ".jpg")
            print("./crop2/" + person_index + "/" + str(count) + ".jpg")
            img = img / 255.0
            img = resize(img, (img_width, img_height))
            x = img.reshape((1, img_width, img_height, 3))
            label = np.argmax(model.predict(x), axis=1)
            # time.sleep(0.5)
            '''
            if label[0] == 0:
                write_log(person_index + "_v1.txt", "0")
            elif label[0] == 1:
                write_log(person_index + "_v1.txt", "2")
            elif label[0] == 2:
                write_log(person_index + "_v1.txt", "1")
            '''
            if label[0] == 0:
                prev = "10"
                write_log(person_index + "_v2.txt", "10")
            elif label[0] == 1:
                prev = "8"
                write_log(person_index + "_v2.txt", "8")
        else:
            write_log(person_index + "_v2.txt", prev)
    else:
        write_log(person_index + "_v2.txt", each)
    count += 1
read_file.close()
'''
for filename in dirs:
    f_dir = os.path.join(dir_name, filename)
    img = io.imread(f_dir)
    img = img / 255.0
    img = resize(img, (img_width, img_height))
    x = img.reshape((1, img_width, img_height, 3))
    print(np.argmax(model.predict(x), axis=1))
    label = np.argmax(model.predict(x), axis=1)
for i in sorted(result):
    if i == 0:
        print("C1: " + str(result[i]))
    elif i == 1:
        print("C3: " + str(result[i]))
    elif i == 2:
        print("C2: " + str(result[i]))
'''