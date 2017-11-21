# -*- coding:utf-8 -*-
__author__ = 'Merle'
import math

import numpy as np


def poseEstimation(img, pt2d):

    height, width, nChannels = img.shape

    LeftVis = np.concatenate((np.arange(9), np.arange(31)+17, np.array([48, 60, 64, 54])))
    RightVis = np.concatenate((np.arange(9)+8, np.arange(31)+17, np.array([48, 60, 64, 54])))

    pt3d = np.load('pt3d.npy')
    # Coarse Pose Estimate
    phil, gammal, thetal, t3d, f = PoseEstimation(pt2d[:, LeftVis], pt3d[:, LeftVis])
    phir, gammar, thetar, t3d, f = PoseEstimation(pt2d[:, RightVis], pt3d[:, RightVis])
    if abs(gammal) > abs(gammar):
        phi = phil
        gamma = gammal
        theta = thetal
    else:
        phi = phir
        gamma = gammar
        theta = thetar

    return([phi, gamma, theta])


def PoseEstimation(pt2d, pt3d):
# 参照论文Optimum Fiducials Under Weak Perspective Projection，使用弱透视投影
    # 减均值，排除t，便于求出R
    pt2dm = np.zeros(pt2d.shape)
    pt3dm = np.zeros(pt3d.shape)
    pt2dm[0, :] = pt2d[0, :]-np.mean(pt2d[0, :])
    pt2dm[1, :] = pt2d[1, :]-np.mean(pt2d[1, :])
    pt3dm[0, :] = pt3d[0, :]-np.mean(pt3d[0, :])
    pt3dm[1, :] = pt3d[1, :]-np.mean(pt3d[1, :])
    pt3dm[2, :] = pt3d[2, :]-np.mean(pt3d[2, :])
    # 最小二乘方法计算R
    R1 = np.dot(np.dot(np.mat(np.dot(pt3dm, pt3dm.T)).I, pt3dm), pt2dm[0, :].T)
    R2 = np.dot(np.dot(np.mat(np.dot(pt3dm, pt3dm.T)).I, pt3dm), pt2dm[1, :].T)
    # 计算出f
    f = (math.sqrt(R1[0, 0]**2+R1[0, 1]**2+R1[0, 2]**2)+math.sqrt(R2[0, 0]**2+R2[0, 1]**2+R2[0, 2]**2))/2
    R1 = R1/f
    R2 = R2/f
    R3 = np.cross(R1, R2)
    # SVD 分解，重构
    U, s, V = np.linalg.svd(np.concatenate((R1, R2, R3), axis=0), full_matrices=True)
    R = np.dot(U, V)
    R1 = R[0, :]
    R2 = R[1, :]
    R3 = R[2, :]
    # 使用旋转矩阵R恢复出旋转角度
    phi = math.atan(R2[0, 2]/R3[0, 2])
    gamma = math.atan(-R1[0, 2]/(math.sqrt(R1[0, 0]**2+R1[0, 1]**2)))
    theta = math.atan(R1[0, 1]/R1[0, 0])
    # 使用R重新计算旋转平移矩阵，求出t
    pt3d = np.row_stack((pt3d, np.ones((1, pt3d.shape[1]))))
    R1_orig = np.dot(np.dot(np.mat(np.dot(pt3d, pt3d.T)).I, pt3d), pt2d[0, :].T)
    R2_orig = np.dot(np.dot(np.mat(np.dot(pt3d, pt3d.T)).I, pt3d), pt2d[1, :].T)

    t3d = np.array([R1_orig[0, 3], R2_orig[0, 3], 0]).reshape((3, 1))
    return(phi, gamma, theta, t3d, f)
