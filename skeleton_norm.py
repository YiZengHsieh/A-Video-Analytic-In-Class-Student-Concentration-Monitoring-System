#!/bin/env python3
import csv
import math
import numpy as np
import numpy.linalg as la
import copy
import os
import random
import time
from multiprocessing import Process
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optparse import OptionParser
from scipy import signal
from fileIO import *
import sys
sys.path.insert(0, '../')
# from message import *
from dataProcess import *
Height = 512
Width = 512
# use this factor to normalize sekelton
scale_factor_for_skeleton = 50


class VideoJointNorm:
    name = dict([["Nose", 0], ["Neck", 1], ["RShoulder", 2], ["RElbow", 3], ["RWrist", 4], ["LShoulder", 5], ["LElbow", 6], ["LWrist", 7], ["RHip", 8], ["RKnee", 9],
                 ["RAnkle", 10], ["LHip", 11], ["LKnee", 12], ["LAnkle", 13], ["REye", 14], ["LEye", 15], ["REar", 16], ["LEar", 17], ["Bkg", 18]])
    data_shift = 1
    data_per_elements = 3

    def __init__(self, data, is_random=True):
        # TODO data and data_normalized should change
        self.data = np.array(data)
        self.raw_data = np.array(data)
        self.data_ori = self.readFromRaw(np.array(data))
        # print('last idx', self.data_ori[-1][0])

        # important randomzie data
        if is_random:
            random.shuffle(self.data_ori)

        self.data_normalized = self.normalize()
        # runtimeMessage("Info", "Data len: " + str(len(self.data_ori)) + ", " + str(len(self.data_normalized)))

    def drawSkeleton(self, skeleton, fig_idx=1, normalize_flg=False):
        fig = plt.figure(fig_idx)
        fig.canvas.set_window_title("frame number :" + str(fig_idx))

        arm_list_name = ["LWrist", "LElbow", "LShoulder",
                         "Neck", "RShoulder", "RElbow", "RWrist"]
        leg_list_name = ["LAnkle", "LKnee", "LHip",
                         "Neck", "RHip", "RKnee", "RAnkle"]

        # arm_list = np.array([skeleton[self.name["LShoulder"]], skeleton[self.name["Neck"]], skeleton[self.name["RShoulder"]]])
        # leg_list = np.array([skeleton[self.name["LHip"]], skeleton[self.name["Neck"]], skeleton[self.name["RHip"]]])

        # arm_list = np.array([skeleton[self.name["LElbow"]], skeleton[self.name["LShoulder"]], skeleton[self.name["Neck"]], skeleton[self.name["RShoulder"]], skeleton[self.name["RElbow"]]])
        # leg_list = np.array([skeleton[self.name["LKnee"]], skeleton[self.name["LHip"]], skeleton[self.name["Neck"]], skeleton[self.name["RHip"]], skeleton[self.name["RKnee"]]])

        # arm_list = np.array([skeleton[self.name["LWrist"]], skeleton[self.name["LElbow"]], skeleton[self.name["LShoulder"]], skeleton[self.name["Neck"]], skeleton[self.name["RShoulder"]], skeleton[self.name["RElbow"]], skeleton[self.name["RWrist"]]])
        # leg_list = np.array([skeleton[self.name["LAnkle"]], skeleton[self.name["LKnee"]], skeleton[self.name["LHip"]], skeleton[self.name["Neck"]], skeleton[self.name["RHip"]], skeleton[self.name["RKnee"]], skeleton[self.name["RAnkle"]]])

        # print("skl->", skeleton)

        arm_list = []
        for each_joint in arm_list_name:
            # print(skeleton[self.name[each_joint]])
            if (not normalize_flg and skeleton[self.name[each_joint]][0] != 0) or (normalize_flg and skeleton[self.name[each_joint]][0] >= -150):
                arm_list.append(skeleton[self.name[each_joint]])

        leg_list = []
        for each_joint in leg_list_name:
            if (not normalize_flg and skeleton[self.name[each_joint]][0] != 0) or (normalize_flg and skeleton[self.name[each_joint]][0] >= -150):

                # if skeleton[self.name[each_joint]][0] != 0  and
                # skeleton[self.name[each_joint]][0] <= -150:
                leg_list.append(skeleton[self.name[each_joint]])

        # check if you wath to change this function
        # if skeleton[self.name['Nose']][0] != 0:
        #     tmp_point_list = np.append(np.array([[0,0]]), [skeleton[self.name['Nose']]], axis=0)
        #     print(tmp_point_list)
        #     print(tmp_point_list[:,0], tmp_point_list[:,1])
        #     plt.plot(tmp_point_list[:,0], tmp_point_list[:,1],color='g')

        # print(arm_list)
        arm_list = np.array(arm_list)
        leg_list = np.array(leg_list)
        # print("arm---- ",arm_list, "\nleg---", leg_list)

        plt.plot(arm_list[:, 0], arm_list[:, 1], color='b')
        plt.plot(leg_list[:, 0], leg_list[:, 1], color='y')

    # draw with normalize fig use for cnn later
    def drawSkeletonTwo(self, data_idx, fig_idx=1):
        fig = plt.figure(fig_idx)
        fig.set_size_inches(5, 5)
        fig.canvas.set_window_title("frame number :" + str(fig_idx))

        arm_list_name = ["LWrist", "LElbow", "LShoulder",
                         "Neck", "RShoulder", "RElbow", "RWrist"]
        leg_list_name = ["LHip", "Neck", "RHip"]
        head_list_name = ["LEar", "LEye", "Nose", "REye", "REar"]

        skeleton = self.data[data_idx][2]

        # print("data ori ", self.data_ori[data_idx][2])
        # print("data ", self.data[data_idx][2])
        isEmpty_list = self.data_ori[data_idx][2] == [0.0]
        # print(isEmpty_list)
        arm_list = []
        for each_joint in arm_list_name:
            # print(skeleton[self.name[each_joint]])
            # if (not normalize_flg and skeleton[self.name[each_joint]][0] !=
            # 0) or (normalize_flg and skeleton[self.name[each_joint]][0] >=
            # -150):
            if not isEmpty_list[self.name[each_joint]][0] or not isEmpty_list[self.name[each_joint]][1]:
                arm_list.append(skeleton[self.name[each_joint]])

        leg_list = []
        for each_joint in leg_list_name:
            # if (not normalize_flg and skeleton[self.name[each_joint]][0] !=
            # 0) or (normalize_flg and  skeleton[self.name[each_joint]][0] >=
            # -150):

            # if skeleton[self.name[each_joint]][0] != 0  and skeleton[self.name[each_joint]][0] <= -150:
            # print(not isEmpty_list[self.name[each_joint]][0])
            if not isEmpty_list[self.name[each_joint]][0] or not isEmpty_list[self.name[each_joint]][1]:
                leg_list.append(skeleton[self.name[each_joint]])

        head_list = []
        for each_joint in head_list_name:
            # if (not normalize_flg and skeleton[self.name[each_joint]][0] !=
            # 0) or (normalize_flg and  skeleton[self.name[each_joint]][0] >=
            # -150):

            # if skeleton[self.name[each_joint]][0] != 0  and skeleton[self.name[each_joint]][0] <= -150:
            # print(not isEmpty_list[self.name[each_joint]][0])
            if not isEmpty_list[self.name[each_joint]][0] or not isEmpty_list[self.name[each_joint]][1]:
                head_list.append(skeleton[self.name[each_joint]])

        arm_list = np.array(arm_list)
        leg_list = np.array(leg_list)
        head_list = np.array(head_list)
        # print("arm---- ",arm_list, "\nleg---", leg_list)

        plt.plot(arm_list[:, 0], arm_list[:, 1], color='g')
        plt.plot(leg_list[:, 0], leg_list[:, 1], color='b')

        # draw head check if need to be modify if function touched
        if self.data[data_idx][1][self.name['Nose']]:
            # drawing
            plt.plot(head_list[:, 0], head_list[:, 1], color='r')
            tmp_point_list = np.append(
                np.array([[0, 0]]), [skeleton[self.name['Nose']]], axis=0)
            plt.plot(tmp_point_list[:, 0], tmp_point_list[:, 1], color='r')


    def getHeight(self, skeleton):
        # if skeleton[self.name["LHip"]][0] == 0 or skeleton[self.name["RHip"]][0] == 0:
        #     return 0
        # input array must be numpy array
        # print(skeleton)
        middle_Hip = 0.5 * \
            (skeleton[self.name["LHip"]] + skeleton[self.name["RHip"]])
        # print("middle hip->", middle_Hip)
        # print("Neck->", skeleton[self.name["Neck"]])
        return la.norm(skeleton[self.name["Neck"]] - middle_Hip)
        # return np.norm(skeleton[self.name["Neck"]] - 0.5 *
        # (self.getJoint(frame_idx, self.name["RHip"]) +
        # self.getJoint(frame_idx, self.name["LHip"])))

    def getShoulderWidth(self, skeleton):
        # if skeleton[self.name["LShoulder"]][0] == 0 or skeleton[self.name["RShoulder"]][0] == 0:
        #     return 0
        return la.norm(skeleton[self.name["LShoulder"]] - skeleton[self.name["RShoulder"]])

    def checkSignificantJoint(self, confidence):
        significant_joint = ["Neck", "RShoulder", "LShoulder", "RHip", "LHip"]
        # print("confidence->", (np.array(confidence)*10).astype(int))
        for joint_name in significant_joint:
            if confidence[self.name[joint_name]] == 0:
                return False
        return True

        # for each_confidence in confidence:
        #     # print("each_confidence->", each_confidence)
        #     if each_confidence == 0:
        #         return False
        # return True
        # for joint_name in self.:
        #     confidence_idx = getIdx(self.name[joint_name]) + 2 #2 for (x, y, c)
        #     if self.data[frame_idx][confidence_idx] == 0:
        #         return False
    # @staticmethod
    def readFromRaw(self, data):
        # print(data[0])
        tmp_list = []
        # runtimeMessage("Info", "data len" + str(len(data)))
        for each_skleteon in data:
            # [idx, [confidence_list]]
            # TODO try to find x,y with the following method
            tmp_x_list = [each_joint_x for each_joint_x in each_skleteon[1::3]]
            tmp_y_list = [each_joint_y for each_joint_y in each_skleteon[2::3]]
            # print("ziping", list(zip(tmp_x_list, tmp_y_list)))
            # for each_joint in each_skleteon[1::3]:
            #     tmp_joint = []
            #     tmp_joint.append()
            tmp_skleteon = [each_skleteon[0], [each_confidence for each_confidence in each_skleteon[
                3::3]], np.array(list(zip(tmp_x_list, tmp_y_list)))]
            if not self.checkSignificantJoint(tmp_skleteon[1]):
                # print('idx:{}'.format(each_skleteon[0]))
                # print("skip")
                continue
            tmp_list.append(tmp_skleteon)
        # [print("tmp_list->", i) for i in tmp_list]
        # print("tmp_skl-> ", tmp_list)
        return tmp_list
    # @staticmethod

    def normalize(self):
        self.data = []
        # print("before norm",len(self.data_ori))
        # print("idx of data_ori {}".format(self.data_ori[-1][0]))
        for each_data in self.data_ori:
            # print("each data->", each_data)
            if not self.checkSignificantJoint(each_data[1]):
                # print("skip")
                # print('idx: ', each_data[0])
                continue
            else:
                # print('idx: ', each_data[0])

                tmp_height = self.getHeight(np.array(each_data[2]))
                tmp_shoulderWidth = self.getShoulderWidth(np.array(each_data[2]))

                if tmp_height == 0 and tmp_shoulderWidth == 0:
                    # print("H vs S", tmp_height, ", ", tmp_shoulderWidth)
                    #
                    # print(each_data[2].astype(int))
                    # input("test")
                    # print('idx: ', each_data[0])

                    continue
                tmp_element = each_data[0:2]

                tmp_element.append((each_data[2] - each_data[2][self.name["Neck"]]) / (tmp_shoulderWidth * 1.2 / 50))

                self.data.append(tmp_element)

                # show debug info
                # print("idx->", each_data[0])
                # print("ori data-> ", np.array(each_data[2]).astype(int))
                # print("tmp data-> ", np.array(tmp_element[2]).astype(int))
                # print("H vs S", tmp_height, ", ", tmp_shoulderWidth)
                # if each_data[0] == 193:
                #     input("Enter:")
        # print(len(self.data))
        # print('idx:{}'.format(self.data[-1][0]))
        return self.data
    def draw_once(self, data_idx, pause_time = 0.05, is_clf = False):
        self.drawSkeletonTwo(data_idx, 99)
        unit_length = scale_factor_for_skeleton

        x = 0 - 1.5 * unit_length, 0 + 1.5 * unit_length
        y = 0 - 1.5 * unit_length, 0 + 3.0 * unit_length

        fig = plt.figure(99, figsize=(1, 1))

        fig.canvas.set_window_title('Data index:{}'.format(data_idx))

        plt.ylim(y)
        plt.xlim(x)

        plt.autoscale(False)
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.ion()

        plt.pause(pause_time)
        # time.sleep(0.1)
        # input("Press Enter To Continue")
        if is_clf:
            plt.clf()


    def draw(self, start_idx=0, end_idx=None, path_prefix="./output/", numberOfData=100):
        counter = 0
        if path_prefix != None:
            # runtimeMessage("Info", "Output folder: " + path_prefix)
            if not os.path.isdir(path_prefix):
                os.makedirs(path_prefix)
            if path_prefix[-1] != '/':
                path_prefix += '/'

        for data_idx, each_frame in enumerate(self.data):

            # runtimeMessage("Info", "Frame idx: {: 6.0f}({: 4.0%})".format(each_frame[0], data_idx / len(self.data_ori)), flag_con_run=True)

            if each_frame[0] <= start_idx or (False if end_idx == None else each_frame[0] >= end_idx):
                # if int(each_frame[0]) not in in_frame:
                # print(each_frame[0])
                continue
            else:
                counter += 1
                # print("Draw frame data ", each_frame[0])
                self.drawSkeletonTwo(data_idx, 2)
                # self.drawSkeleton(self.data_ori[data_idx][2], 1, False)


            if path_prefix == None:
                continue

            # method 2 session#######################################

            unit_length = scale_factor_for_skeleton
            # unit_length = self.getShoulderWidth(np.array(each_frame[2])) * 2

            x = 0 - 1.0 * unit_length, 0 + 1.0 * unit_length
            y = 0 - 1.0 * unit_length, 0 + 3.0 * unit_length

            plt.figure(2, figsize=(1, 1))

            # old
            # plt.ylim(y)
            # plt.xlim(x)
            # print(x, y)

            #########################################################
            # else
            #########################################################

            # plt.figure(2)
            # plt.ylim([-50, 150])
            # plt.xlim([-100, 100])
            #########################################################

            plt.autoscale(False)
            # old
            plt.axis('equal')
            plt.gca().invert_yaxis()
            plt.axis('off')

            # output fig data
            plt.savefig(path_prefix + "skeleton-" +
                        str(counter) + '.jpg', pad_inches=0)
            # plt.savefig(path_prefix + "skeleton-" + str(each_frame[0]) + '.png', bbox_inches='tight', pad_inches=0)
            plt.clf()


            if numberOfData != 0 and counter > numberOfData:
                break

        # runtimeMessage("Info", "Len: " + str(len(self.data_ori)) + ", Count: " + str(counter))

    def _calcAngle(self, data_idx, joint_list):
        # warp calc_angle
        # no warp
        for joint_idx in joint_list:
            if self.data_ori[data_idx][2][joint_idx][0] == 0:
                return -200
        return cals_angle_norm(self.data, data_idx, joint_list)
    def _calc_x_angle(self, data_idx, joint_list):
        # warp calc_angle
        # no warp
        for joint_idx in joint_list:
            if self.data_ori[data_idx][2][joint_idx][0] == 0:
                return -200
        return cals_angle_horizontal(self.data, data_idx, joint_list)
    def _calcLength(self, data_idx, joint_list):
        # calc length with la.norm
        # no warp
        # print(joint_list)
        for joint_idx in joint_list:
            if self.data_ori[data_idx][2][joint_idx][0] == 0:
                return -200
        vector_pair = self.data[data_idx][2][
            joint_list[0]] - self.data[data_idx][2][joint_list[1]]
        return la.norm(vector_pair)
    # TODO CALC BODY LENGTH
    def _calcbodyLength(self, data_idx):
        # calc length with la.norm
        # no warp
        # print(joint_list)
        # joint_list = [self.name['Neck'], self.name['RAnkle'], self.name['LAnkle']]
        ankle_middle = -200
        if self.data_ori[data_idx][2][self.name['RAnkle']][1] == 0 and 0 == self.data_ori[data_idx][2][self.name['RAnkle']][1]:
            return -200
        elif self.data_ori[data_idx][2][self.name['RAnkle']][1] == 0:
            ankle_middle = self.data_ori[data_idx][2][self.name['LAnkle']][1]
        elif self.data_ori[data_idx][2][self.name['LAnkle']][1] == 0:
            ankle_middle = self.data_ori[data_idx][2][self.name['RAnkle']][1]
        else:
            ankle_middle = (self.data_ori[data_idx][2][self.name['RAnkle']][1] + self.data_ori[data_idx][2][self.name['LAnkle']][1]) * 0.5

        return ankle_middle
    def _calcbodyratio(self, data_idx):
        # calc length with la.norm
        # no warp
        # print(joint_list)
        # joint_list = [self.name['Neck'], self.name['RAnkle'], self.name['LAnkle']]
        ankle_middle = -200
        shoudler_width = self.getShoulderWidth(np.array(self.data[data_idx][2]))
        print(shoudler_width)
        if self.data_ori[data_idx][2][self.name['RAnkle']][1] == 0 and 0 == self.data_ori[data_idx][2][self.name['RAnkle']][1]:
            return -200
        elif self.data_ori[data_idx][2][self.name['RAnkle']][1] == 0:
            ankle_middle = self.data_ori[data_idx][2][self.name['LAnkle']][1]
        elif self.data_ori[data_idx][2][self.name['LAnkle']][1] == 0:
            ankle_middle = self.data_ori[data_idx][2][self.name['RAnkle']][1]
        else:
            ankle_middle = (self.data_ori[data_idx][2][self.name['RAnkle']][1] + self.data_ori[data_idx][2][self.name['LAnkle']][1]) * 0.5

        return ankle_middle / shoudler_width
    def _calcdistance(self, y_pos, y_center, unit_length):
        # y_pos = self.data[data_idx][2][self.name['RWrist']][y_axis]
        # y_center = self.data[data_idx][2][self.name['RShoulder']][y_axis]
        if y_pos > y_center:
            # below shoulder
            center_dist = y_pos - y_center
            return center_dist if center_dist < unit_length else unit_length
        else:
            # y_pos < y_center
            # upon shoudler
            center_dist = y_center - y_pos
            return -center_dist if center_dist < unit_length else -unit_length

    def genPredData(self, class_idx=0, total_class=6):
        # predefine angle order list
        arm_list_name = ["LWrist", "LElbow", "LShoulder",
                         "Neck", "RShoulder", "RElbow", "RWrist"]
        leg_list_name = ["LAnkle", "LKnee", "LHip",
                         "Neck", "RHip", "RKnee", "RAnkle"]
        joint_list_name = (arm_list_name + leg_list_name)
        joint_list_name.remove("Neck")
        joint_list_name.remove("Neck")

        total_data = len(self.data)
        output_data = []
        idx_list = []
        print('len of self.data', len(self.data))
        for data_idx, each_frame in enumerate(self.data):

            tmp_data = []

            # TODO mark if the value is not avalibalie
            # angle
            # arm_list
            for arm_joint_idx in range(0, len(arm_list_name) - 2):
                tmp_data.append(self._calcAngle(data_idx, [self.name[
                                joint_idx] for joint_idx in arm_list_name[arm_joint_idx:arm_joint_idx + 3]]))
                # tmp_data.append(cals_angle_norm(self.data, data_idx, [self.name[joint_idx] for joint_idx in arm_list_name[arm_joint_idx:arm_joint_idx+3]]))
            # print("angle: {}".format(tmp_data))
            # input()
            # break

            # leg list
            for leg_joint_idx in range(0, len(leg_list_name) - 2):
                tmp_data.append(self._calcAngle(data_idx, [self.name[
                                joint_idx] for joint_idx in leg_list_name[leg_joint_idx:leg_joint_idx + 3]]))
                # tmp_data.append(cals_angle_norm(self.data, data_idx, [self.name[joint_idx] for joint_idx in leg_list_name[leg_joint_idx:leg_joint_idx+3]]))

            # method 3
            y_axis = 1
            unit_length = self.getHeight(np.array(each_frame[2])) * 2
            # unit_length = self.getHeight(np.array(each_frame[2]))
            # unit_length = self.getShoulderWidth(np.array(each_frame[2]))
            # RShoulder
            y_pos = self.data[data_idx][2][self.name['RWrist']][y_axis]
            y_center = self.data[data_idx][2][self.name['RShoulder']][y_axis]
            tmp_data.append(self._calcdistance(y_pos, y_center, unit_length))

            # LShoulder
            y_pos = self.data[data_idx][2][self.name['LWrist']][y_axis]
            y_center = self.data[data_idx][2][self.name['LShoulder']][y_axis]
            tmp_data.append(self._calcdistance(y_pos, y_center, unit_length))

            # RKnee
            y_pos = self.data[data_idx][2][self.name['RWrist']][y_axis]
            y_center = self.data[data_idx][2][self.name['RKnee']][y_axis]
            tmp_data.append(self._calcdistance(y_pos, y_center, unit_length))

            # LKnee
            y_pos = self.data[data_idx][2][self.name['LWrist']][y_axis]
            y_center = self.data[data_idx][2][self.name['LKnee']][y_axis]
            tmp_data.append(self._calcdistance(y_pos, y_center, unit_length))

            # end of method

            # LNose reletive position
            # y_pos = self.data[data_idx][2][self.name['Nose']][y_axis]
            # y_center = self.data[data_idx][2][self.name['Neck']][y_axis]
            # tmp_data.append(self._calcdistance(y_pos, y_center, unit_length))


            # leg length%
            # tmp_data.append(self._calcLength(data_idx, [self.name['RHip'], self.name['RKnee']]))
            # tmp_data.append(self._calcLength(data_idx, [self.name['LHip'], self.name['LKnee']]))

            #body height
            # tmp_data.append(self._calcbodyLength(data_idx))

            # body limbs angle
            tmp_data.append(self._calc_x_angle(data_idx, [self.name['LWrist'], self.name['LShoulder']]))
            tmp_data.append(self._calc_x_angle(data_idx, [self.name['RWrist'], self.name['RShoulder']]))

            tmp_data.append(self._calc_x_angle(data_idx, [self.name['LKnee'], self.name['LAnkle']]))
            tmp_data.append(self._calc_x_angle(data_idx, [self.name['RKnee'], self.name['RAnkle']]))


            output_data.append(tmp_data)
            idx_list.append(int(each_frame[0]))

        # print("len of dataset", len(output_data))
        # print("len of dim", len(output_data[0]))
        idx_list = np.array(idx_list) - 1

        return idx_list, output_data

    def outputNetworkData(self, class_idx=0, total_class=6, numberOfData = 200, output_file='./output/networkdata.csv'):
        print("Class idx:{}".format(class_idx))
        if numberOfData == 0:
            numberOfData = None
        # output csv data for training in nural network
        # the data contain each joints angle, length, True if have value

        # predefine angle order list
        arm_list_name = ["LWrist", "LElbow", "LShoulder",
                         "Neck", "RShoulder", "RElbow", "RWrist"]
        leg_list_name = ["LAnkle", "LKnee", "LHip",
                         "Neck", "RHip", "RKnee", "RAnkle"]
        joint_list_name = (arm_list_name + leg_list_name)
        joint_list_name.remove("Neck")
        joint_list_name.remove("Neck")
        # joint_list_name = copy.copy(arm_list_name)
        # joint_list_name.extend(copy.copy(leg_list_name))
        # joint_list_name.remove("Neck")
        # print(joint_list_name)
        # print(arm_list_name)
        # print(leg_list_name)
        # input()

        total_data = len(self.data)
        output_data = []
        # for data_idx, each_frame in enumerate(self.data[:numberOfData]):
        # for data_idx, each_frame in enumerate(self.data):
        for _ in range(numberOfData):
            data_idx = random.randrange(total_data)
            each_frame = self.data[data_idx]


            tmp_data = []
            # print frame detial
            # print(each_frame[i] for i in [0,1,2])
            # print("frame idx:{}, \nframe confidence: {}\nframe data: \n".format(
            #     each_frame[0], each_frame[1]))
            # [print(each_frame[2][joint_idx]) for joint_idx in [self.name[joint_idx] for joint_idx in joint_list_name]]

            # TODO mark if the value is not avalibalie
            # angle
            # arm_list
            for arm_joint_idx in range(0, len(arm_list_name) - 2):
                tmp_data.append(self._calcAngle(data_idx, [self.name[
                                joint_idx] for joint_idx in arm_list_name[arm_joint_idx:arm_joint_idx + 3]]))
                # tmp_data.append(cals_angle_norm(self.data, data_idx, [self.name[joint_idx] for joint_idx in arm_list_name[arm_joint_idx:arm_joint_idx+3]]))
            # print("angle: {}".format(tmp_data))
            # input()
            # break

            # leg list
            for leg_joint_idx in range(0, len(leg_list_name) - 2):
                tmp_data.append(self._calcAngle(data_idx, [self.name[
                                joint_idx] for joint_idx in leg_list_name[leg_joint_idx:leg_joint_idx + 3]]))
                # tmp_data.append(cals_angle_norm(self.data, data_idx, [self.name[joint_idx] for joint_idx in leg_list_name[leg_joint_idx:leg_joint_idx+3]]))

            # method 3
            y_axis = 1
            unit_length = self.getHeight(np.array(each_frame[2])) * 2
            # unit_length = self.getHeight(np.array(each_frame[2]))
            # unit_length = self.getShoulderWidth(np.array(each_frame[2]))
            # RShoulder
            y_pos = self.data[data_idx][2][self.name['RWrist']][y_axis]
            y_center = self.data[data_idx][2][self.name['RShoulder']][y_axis]
            tmp_data.append(self._calcdistance(y_pos, y_center, unit_length))

            # LShoulder
            y_pos = self.data[data_idx][2][self.name['LWrist']][y_axis]
            y_center = self.data[data_idx][2][self.name['LShoulder']][y_axis]
            tmp_data.append(self._calcdistance(y_pos, y_center, unit_length))

            # RKnee
            y_pos = self.data[data_idx][2][self.name['RWrist']][y_axis]
            y_center = self.data[data_idx][2][self.name['RKnee']][y_axis]
            tmp_data.append(self._calcdistance(y_pos, y_center, unit_length))

            # LKnee
            y_pos = self.data[data_idx][2][self.name['LWrist']][y_axis]
            y_center = self.data[data_idx][2][self.name['LKnee']][y_axis]
            tmp_data.append(self._calcdistance(y_pos, y_center, unit_length))

            # end of method

            # LNose reletive position
            # y_pos = self.data[data_idx][2][self.name['Nose']][y_axis]
            # y_center = self.data[data_idx][2][self.name['Neck']][y_axis]
            # tmp_data.append(self._calcdistance(y_pos, y_center, unit_length))


            # leg length%
            # tmp_data.append(self._calcLength(data_idx, [self.name['RHip'], self.name['RKnee']]))
            # tmp_data.append(self._calcLength(data_idx, [self.name['LHip'], self.name['LKnee']]))

            #body height
            # tmp_data.append(self._calcbodyLength(data_idx))

            # body limbs angle
            tmp_data.append(self._calc_x_angle(data_idx, [self.name['LWrist'], self.name['LShoulder']]))
            tmp_data.append(self._calc_x_angle(data_idx, [self.name['RWrist'], self.name['RShoulder']]))

            # %
            tmp_data.append(self._calc_x_angle(data_idx, [self.name['LKnee'], self.name['LAnkle']]))
            tmp_data.append(self._calc_x_angle(data_idx, [self.name['RKnee'], self.name['RAnkle']]))




            # print("vector: {}".format(tmp_data[:]))
            # input()

            # append class number
            output_class_y = np.zeros(total_class)
            output_class_y[class_idx] = 1

            # 1 0 0 0 0
            tmp_data.extend(output_class_y)
            # 0
            # tmp_data.append(class_idx)

            output_data.append(tmp_data)

            # draw skeleton
            # self.draw_once(data_idx, is_clf = True)
            # input("Press Enter To Continue")
        # [print(i) for i in output_data]
        print("len of dataset", len(output_data))
        print("len of dim", len(output_data[0]))



        # extract to three trunck
        three_mixed_fold = True

        # stay in one trunck
        # three_mixed_fold = False
        if three_mixed_fold == True:
            if not os.path.isdir(output_file):
                os.makedirs(output_file)

            len_of_data = int(len(output_data) / 3)
            print("Random Data enable")
            print("len of data:{}".format(len_of_data))
            np.random.shuffle(output_data)
            writecsv(os.path.join(output_file, "group_0"), output_data[:len_of_data])
            writecsv(os.path.join(output_file, "group_1"), output_data[len_of_data:len_of_data * 2])
            writecsv(os.path.join(output_file, "group_2"), output_data[len_of_data * 2:])
            print('WARRANTIES: It\'s not a normal version!')
        else:
            writecsv(output_file, output_data)

        # input("Enter")
        return output_data


def makeFileList(root_path, file_name):
    folder_list = os.listdir(root_path)
    folder_list.sort()
    rst = [os.path.join(root_path, each_folder, file_name)
           for each_folder in folder_list]
    return [folder_list, rst]
# use in group folder with multiples csvs
# def makeFileList(root_path, file_name):
#     # folder_list = os.listdir(root_path)
#     print(root_path)
#
#     # for each_folder in folder_list:
#     # for each_file in os.listdir(each_folder):
#
#     # print(os.listdir(root_path))
#     tmp_file_list = []
#     for each_file in os.listdir(root_path):
#         # print(each_file)
#         if each_file.endswith(".csv"):
#             tmp_file_list.append(os.path.join(root_path, each_file))
#     return(tmp_file_list)
#     # print(tmp_file_list)
#
#     # rst = [os.path.join(root_path, each_folder, file_name) for each_folder in folder_list]
#     # print(rst)
#     # return [folder_list, rst]


if __name__ == '__main__':

    # setup cli
    parser = OptionParser(usage='Use batchmail -h to see more info')
    parser.add_option("-j", "--Joint-csv", dest="joint_file",
                      help="Input Joint file", action="store")
    parser.add_option("-d", "--folder_path", dest="folder_path",
                      help="Input Joint folder path", action="store")
    parser.add_option("-s", "--start", dest="idx_start",
                      help="Start index of csv file", type="int", action="store")
    parser.add_option("-e", "--end", dest="idx_end",
                      help="End index of csv file", type="int", action="store")
    parser.add_option("-o", "--output-folder", dest="out_folder",
                      help="Output folder", type="str", action="store")
    parser.add_option("-z", "--output-network-data", dest="output_network_data",
                      help="Output netwowrk data file", type="str", default="./output/networkdata.csv", action="store")
    parser.add_option("-n", "--output-network-data-flag", dest="network_data_flag",
                      help="Output network data flag", default=False, action="store_true")
    parser.add_option("-r", "--disable-random", dest="is_random",
                      help="Disable random", default=True, action="store_false")
    parser.add_option("-x", "--x-size", dest="input_size",
                      help="input size", type="int", default=46, action="store")
    parser.add_option("-y", "--y-size", dest="output_class_size",
                      help="output size", type="int", default=6, action="store")
    parser.add_option("-m", "--max-output-number", dest="max_output_number",
                      help="Max class size", type="int", default=0, action="store")

    # parser.add_option("-l", "--mail-list", dest="mail_list",
    #                   help="The mail list you wish to send", action="store")
    # parser.add_option("-g", "--generate-config", dest="gerenate_config",
    #                   help="Generate template config file", default=False, action="store_true")
    # parser.add_option("-f", "--file", dest="file_list",
    # help="Generate template config file", default=[], action="append")

    # read arg to var
    (options, args) = parser.parse_args()
    if not options.joint_file and not options.folder_path:
        # runtimeMessage("Options", "No joint file ")
        exit()

    if options.idx_start:
        idx_start = options.idx_start
        # runtimeMessage("Options", "Start idx from " + str(idx_start))
    else:
        idx_start = 0
    if options.idx_end:
        idx_end = options.idx_end
        # runtimeMessage("Options", "End idx from " + str(idx_end))
    else:
        idx_end = None

    # if options.out_folder:
    #     out_folder = [options.out_folder]
    #     runtimeMessage("Options", "Output folder: " + out_folder)
    # else:
    #     out_folder = ["test"]

    if options.out_folder:
        path_prefix = options.out_folder
        # runtimeMessage("Options", "Output folder: " + path_prefix)
    else:
        path_prefix = None
        out_folder = "test"

    if options.joint_file:
        src_path = [options.joint_file]
    else:
        print("In folder")
        out_folder, src_path = makeFileList(options.folder_path, "Joints.csv")

    flag_network_data = options.network_data_flag
    output_netowrk_data = options.output_network_data

    # data args
    total_class = options.output_class_size
    numberOfData = options.max_output_number
    class_counter = 0
    for path_idx, each_path in enumerate(src_path):
        # read csv file
        print("<Info> Skeleton normalize\n")
        print("Path({}): {}".format(path_idx, each_path))
        raw_joints = readcsv(each_path)
        processed_joints = []
        # print(raw_joints[0])

        # append all joint in array
        # print("\"", idx_start, "\", \"", idx_end,"\"")
        # runtimeMessage("Info", "Idx Start: End" + str(idx_start) + ": " + str(idx_end))

        for idx, eachframe in enumerate(raw_joints[idx_start: idx_end]):
            # for idx, eachframe in enumerate(raw_joints):
            processed_joints.append(eachframe)
        processed_joints = np.array(processed_joints)

        # debug print out joints
        # [print(joint) for joint in processed_joints[0:1]]

        vjn = VideoJointNorm(processed_joints, is_random = options.is_random)

        if flag_network_data:
            vjn.outputNetworkData(class_idx=class_counter, numberOfData=numberOfData,
                                  total_class=total_class, output_file=output_netowrk_data)
            class_counter = class_counter + 1
        else:
            out_folder = output_netowrk_data
            if path_prefix is not None:
                print(out_folder)
                # os.makedirs(os.path.join(path_prefix, out_folder[path_idx]))
                vjn.draw(path_prefix=os.path.join(path_prefix, out_folder[
                         path_idx]), numberOfData=numberOfData)
            else:
                vjn.draw()


            # # method 1
            # # len of vector
            # # arm length
            # for arm_joint_idx in range(0, len(arm_list_name) - 1):
            #     tmp_data.append(self._calcLength(data_idx, [self.name[
            #                     joint_idx] for joint_idx in arm_list_name[arm_joint_idx:arm_joint_idx + 2]]))
            #
            # # print("length: {}".format(tmp_data[10:]))
            # # input()
            # # leg length
            # for leg_joint_idx in range(0, len(leg_list_name) - 1):
            #     tmp_data.append(self._calcLength(data_idx, [self.name[
            #                     joint_idx] for joint_idx in leg_list_name[leg_joint_idx:leg_joint_idx + 2]]))
            #
            # # vector
            # # for joint_ind in arm_list_name.extend(leg_list_name):
            # joint_list = [self.name[joint_idx]
            #               for joint_idx in joint_list_name]
            # for joint_idx in joint_list:
            #     tmp_data.extend(self.data[data_idx][2][joint_idx] if self.data[
            #                     data_idx][1][joint_idx] != 0 else [-200, -200])

            # # method 2
            # # whrist position knee/shoulder
            # # shoulder
            # # use < coz the y axis is inverted in image
            # y_axis = 1
            # tmp_data.append(1 if self.data[data_idx][2][self.name['RWrist']][
            #                 y_axis] < self.data[data_idx][2][self.name['RShoulder']][y_axis] else 0)
            # tmp_data.append(1 if self.data[data_idx][2][self.name['LWrist']][
            #                 y_axis] < self.data[data_idx][2][self.name['LShoulder']][y_axis] else 0)
            # # knee
            # tmp_data.append(1 if self.data[data_idx][2][self.name['RWrist']][
            #                 y_axis] < self.data[data_idx][2][self.name['RKnee']][y_axis] else 0)
            # tmp_data.append(1 if self.data[data_idx][2][self.name['LWrist']][
            #                 y_axis] < self.data[data_idx][2][self.name['LKnee']][y_axis] else 0)

            # print("right")
            # print(self.data[data_idx][2][self.name['RWrist']])
            # print(self.data[data_idx][2][self.name['RShoulder']])
            # print(self.data[data_idx][2][self.name['RKnee']])
            #
            # print("Left")
            # print(self.data[data_idx][2][self.name['LWrist']])
            # print(self.data[data_idx][2][self.name['LShoulder']])
            # print(self.data[data_idx][2][self.name['LKnee']])
            # print(tmp_data[-4:])
