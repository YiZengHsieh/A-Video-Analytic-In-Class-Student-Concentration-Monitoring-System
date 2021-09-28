#!/bin/env python

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

import sys
import os
# disable tf log message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10

from optparse import OptionParser
from sklearn.metrics import confusion_matrix

import csv


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('output_file', './csv',
                           """Directory where to write results.""")
tf.app.flags.DEFINE_string('dataset', './cifar10_train',
                           """Directory where to load data.""")
tf.app.flags.DEFINE_integer('k_index', 0,
                            """Number of index to test.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def print_confusion_matrix(c_matrix, num_of_class):
    # print(",",",".join(label_name))
    csv_writer = open(FLAGS.output_file, 'a')
    label_name = range(num_of_class)
    for idx in range(num_of_class):
        csv_writer.write(","+str(idx))
    csv_writer.write('\n')
    for each_row in range(len(c_matrix)):        
        print(label_name[each_row], ",", ",".join(map(str, c_matrix[each_row])))
        csv_writer.write(str(label_name[each_row])+ ","+ ",".join(map(str, c_matrix[each_row])))
        csv_writer.write('\n')
    print()
# def pritn_confusion_matrix_9(c_matrix, label_name = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']):
#     print(",",",".join(label_name))
#     for each_row in range(len(c_matrix)):
#         print(label_name[each_row] + ",", ",".join(map(str, c_matrix[each_row])))
#     print()


def eval_once(saver, summary_writer, top_k_op, summary_op, labels, pred, flag, num_of_class = 5):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        print(FLAGS.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            eprint("Network restore")
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            eprint('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            # my Args
            real_lab = []
            pred_lab = []
            while step < num_iter and not coord.should_stop():
                predictions, tmp_real_lab, tmp_pred_lab = sess.run([top_k_op, labels, pred])
                true_count += np.sum(predictions)
                step += 1
                real_lab.extend(tmp_real_lab)
                pred_lab.extend(tmp_pred_lab)
            # print(lab)
            eprint(real_lab.count(0))
            eprint(real_lab.count(1))
            eprint(real_lab.count(2))
            eprint(real_lab.count(3))
            eprint(real_lab.count(4))
            eprint(real_lab.count(5))
            eprint(real_lab.count(6))
            eprint(real_lab.count(7))
            csv_writer = open(FLAGS.output_file, 'a')
            text_file = open(FLAGS.output_file + "result.txt", "a")
            if flag:
                for each in pred_lab:
                    text_file.write(str(each) + "\n")
            text_file.close()
            # print(confusion_matrix(real_lab, pred_lab, range(6)))
            cm_data = confusion_matrix(real_lab, pred_lab, range(num_of_class))
            # for each_row in range(len(cm_data)):
            #     print(",".join(map(str, cm_data[each_row])))
            print_confusion_matrix(cm_data, num_of_class)
            # Compute precision @ 1.
            eprint("True Count:{}, step:{}, num of example:{}".format(true_count, step, FLAGS.num_examples))
            precision = true_count / total_sample_count
            print('%s: precision @ 1 ,%.3f' % (datetime.now(), precision))
            csv_writer.write('\n'+str(precision)+'\n\n\n')

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    size_per_file = 2000
    num_classes = 5

    FLAGS.data_dir = FLAGS.dataset

    # FLAGS.num_examples = 2300 * 6 * 3

    eprint(FLAGS.dataset)
    csv_writer = open(FLAGS.output_file, 'a')
    # csv_writer.write("Test"+str(format(options.k_index))+"\n\n")
    if os.path.isfile(FLAGS.dataset):
        FLAGS.data_dir = [FLAGS.dataset]

        # FLAGS.num_examples = size_per_file * num_classes
    else:
        FLAGS.data_dir = []
        training_set_list = []
        testing_set_list = []

        data_file_list = os.listdir(FLAGS.dataset)
        data_file_list.sort()
        for file_idx, each_file in enumerate(data_file_list):
            if file_idx == FLAGS.k_index:
                testing_set_list.append(os.path.join(FLAGS.dataset, each_file))
                continue
            training_set_list.append(os.path.join(FLAGS.dataset, each_file))

        # print(FLAGS.data_dir)
        # print(FLAGS.num_examples)
        print("train")
        # training set
        eprint("Training:{}".format(training_set_list))
        print("Training,{}".format(FLAGS.k_index))
        FLAGS.num_examples = size_per_file * num_classes * len(training_set_list)
        print(FLAGS.num_examples)
        FLAGS.data_dir = training_set_list

    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar10.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # confusion matrix
        pred = tf.argmax(logits, 1)
        # confusion_matrix = tf.confusion_matrix(labels, pred)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op, labels, pred, flag=False, num_of_class=num_classes)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

    # testing set
    eprint("Testing:{}".format(testing_set_list))
    print("Testing,{}".format(FLAGS.k_index))
    FLAGS.num_examples = size_per_file * num_classes * len(testing_set_list)
    print(FLAGS.num_examples)
    FLAGS.data_dir = testing_set_list

    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar10.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # confusion matrix
        pred = tf.argmax(logits, 1)
        # confusion_matrix = tf.confusion_matrix(labels, pred)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op, labels, pred, flag=True, num_of_class=num_classes)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    # cifar10.maybe_download_and_extract()

    # if tf.gfile.Exists(FLAGS.eval_dir):
        # tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    '''
    parser = OptionParser(usage='Use -h to see more infomation')
    parser.add_option("-d", "--dataset", dest="dataset",
                      help="dataset path", type="str", action="store")

    parser.add_option("-i", "--interval", dest="interval",
                      help="Iteration times", type="int", default=5, action="store")
    parser.add_option("-m", "--model-path", dest="model_path",
                      help="Modle path", type="str", default="./model", action="store")

    parser.add_option("-k", "--k-value", dest="k_value",
                      help="k fold", type="int", action="store")
    parser.add_option("-z", "--k-index", dest="k_index",
                      help="k index", type='int', default=-1, action="store")
    parser.add_option("-n", "--num-of-example", dest="num_of_example",
                      help="number of example", type='int', default=15000, action="store")
    parser.add_option("-r", "--rune-once", dest="rune_once",
                      help="rune once", default=True, action="store_false")
    parser.add_option("-o", "--output-file", help="Output file name",
        dest="output_file", type="str", action="store")
    (options, args) = parser.parse_args()
    '''

    tf.app.run()

    # FLAGS.data_dir = ['/tmp/cifar10_new/cifar10_data/cifar-10-batches-bin/data_batch_1.bin', '/tmp/cifar10_new/cifar10_data/cifar-10-batches-bin/data_batch_2.bin',
    #              '/tmp/cifar10_new/cifar10_data/cifar-10-batches-bin/data_batch_3.bin', '/tmp/cifar10_new/cifar10_data/cifar-10-batches-bin/data_batch_4.bin']
    # FLAGS INIT

    # end of file
