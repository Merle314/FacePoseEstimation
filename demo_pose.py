# -*- coding:utf-8 -*-
__author__ = 'Merle'
import math
import os
import time

import numpy as np
from skimage import io

import check_resources as check
import dlib
import facial_feature_detector as feature_detection
import PoseEstimation as PE

this_path = os.path.dirname(os.path.abspath(__file__))
check.check_dlib_landmark_weights()  # 检测dlib参数是否下载，没下载的话下载
image_name = 'zjl.jpg'
image_path = this_path + '/input/' + image_name

time1 = time.time()
print(this_path)
predictor_path = this_path + "/dlib_models/shape_predictor_68_face_landmarks.dat"
print(predictor_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
time2 = time.time()


def demo():
    img = io.imread(image_path)
    # 检测特征点
    time3 = time.time()
    lmarks = feature_detection.get_landmarks(img, detector, predictor)
    time4 = time.time()
    if len(lmarks):
        Pose_Para = PE.poseEstimation(img, lmarks)
        time5 = time.time()
    else:
        print('NO face detected!')
        return 0

    print('Dlib Model Load time is:', time2-time1, 'Lmarks Detect time is:', time4-time3, 'Pose estimation time is:', time5-time4)
    print(np.array(Pose_Para)*180/math.pi)


if __name__ == "__main__":
    demo()
