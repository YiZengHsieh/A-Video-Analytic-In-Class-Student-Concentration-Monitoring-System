# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10
import cifar10_input

from optparse import OptionParser
from sklearn.metrics import confusion_matrix
from PIL import Image

import csv
import cv2
import keras
from shutil import copyfile

# disable tf log message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('output_file', './csv',
                           """Directory where to write results.""")
tf.app.flags.DEFINE_string('filename', './file_name',
                           """file name.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 3,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
NUMBER_OF_CLASS = cifar10_input.NUM_CLASSES




def calc_confusion_matrix(label, pred):
    label = np.array(label)
    pred = np.array(pred)
    print("len: label {}, pred {}".format(len(label), len(pred)))
    # print(pred[:100], label[:100])
    for each_class in range(NUMBER_OF_CLASS):
        # print(pred == each_class)
        print("{} class:{}".format(each_class, np.sum(label == each_class)))
    for each_class in range(NUMBER_OF_CLASS):
        # print(pred == each_class)
        print("{} class:{}".format(each_class, np.sum(pred == each_class)))

    cmatrix = confusion_matrix(pred, label, labels=range(NUMBER_OF_CLASS))
    print(cmatrix)

# [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
# [batch_size]

# def eval_once(saver, summary_writer, top_k_op, pred_lab, true_lab,
# summary_op):
def convert(num):
    if num == 0:
        return "寫筆記/筆電/手機"
    elif num == 1:
        return "舉手"
    elif num == 2:
        return "歪頭"
    elif num == 3:
        return "手撐頭+接電話"
    elif num == 4:
        return "趴睡"
    '''
    if num == 0:
        return "站立"
    elif num == 1:
        return "彎腰"
    elif num == 2:
        return "蹲姿"
    elif num == 3:
        return "坐姿"
    elif num == 4:
        return "吃東西"
    elif num == 5:
        return "舉單手"
    elif num == 6:
        return "舉雙手"
    elif num == 7:
        return "喝水坐"
    elif num == 8:
        return "喝水站"
    elif num == 9:
        return "跌倒"
    '''
    

def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, shape=(
            FLAGS.batch_size, cifar10_input.IMAGE_SIZE, cifar10_input.IMAGE_SIZE, 3), name='images')
        labels = tf.placeholder(tf.int32, shape=(
            FLAGS.batch_size), name='labels')
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data == 'test'
        # images, labels = cifar10.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)

        # testing
        pred_lab = tf.argmax(logits, 1)
        # true_lab = tf.argmax(labels, 1)
        #
        # cmatrix = tf.Variable(labels, name="labels")

        # calc_confusion_matrix(pred_lab, labels)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        # while True:
        #     eval_once(saver, summary_writer, top_k_op, labels, pred_lab, summary_op)
        #     # eval_once(saver, summary_writer, top_k_op, pred_lab, true_lab, summary_op)
        #     if FLAGS.run_once:
        #         break
        #     time.sleep(FLAGS.eval_interval_secs)
        print("In to session")
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split(
                    '/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return
            im = Image.open(FLAGS.filename)
            # im.thumbnail(size, Image.BILINEAR)
            im = im.resize((32, 32), Image.BILINEAR)
            # im.show()
            # input()
            im = (np.array(im))
            # im = im.astype(float)
            # print(im.shape)

            # batch_x = np.zeros(1728).reshape((
            #     FLAGS.batch_size, cifar10_input.IMAGE_SIZE, cifar10_input.IMAGE_SIZE, 3))
            # batch_x = [(im - np.mean(im))/np.var(im)]
            batch_x = tf.image.per_image_standardization(im).eval()

            # print(batch_x)

            batch_x = [batch_x]
            batch_y = [1]

            img_label = pred_lab.eval(feed_dict={images: batch_x, labels: batch_y})
            predictions = sess.run([top_k_op], feed_dict={images: batch_x, labels: batch_y})
            # print("pred:{}".format(logits.eval(feed_dict={images: batch_x, labels: batch_y})))
            # print("pred:{}".format(img_label))

            # feed_dict={self.x: batch_x, self.y: batch_y}


def main(argv=None):  # pylint: disable=unused-argument
    # cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    '''
    parser = OptionParser(usage='Use -h to see more infomation')
    parser.add_option("-i", "--interval", dest="interval",
                      help="Iteration times", type="int", default=5, action="store")
    parser.add_option("-m", "--model-path", dest="model_path",
                      help="Model path", type="str", default="./model", action="store")

    parser.add_option("-n", "--num-of-example", dest="num_of_example",
                      help="number of example", type='int', default=10000, action="store")
    parser.add_option("-r", "--rune-once", dest="num_of_example",
                      help="number of example", default=True, action="store_true")
    parser.add_option("-d", "--data-file", help="data file name", dest="dataFile", type="str", action="store")
    parser.add_option("-c", "--csv-file", help="csv file name", dest="csvFile", type="str", action="store")
    (options, args) = parser.parse_args()
    '''

    tf.app.run()
    # FLAGS INIT
    '''
    tmp_file = options.dataFile
    FLAGS.checkpoint_dir = options.model_path
    FLAGS.eval_dir = 'output/eval'
    FLAGS.eval_interval_secs = options.interval
    FLAGS.num_examples = options.num_of_example
    FLAGS.run_once = True

    FLAGS.batch_size = 1
    '''
    main()
