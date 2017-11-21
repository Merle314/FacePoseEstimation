__author__ = 'Douglas and Iacopo'

import dlib
import os
import numpy as np
import matplotlib.pyplot as plt

def _shape_to_np(shape):
    xy = []
    for i in range(68):
        xy.append((shape.part(i).x, shape.part(i).y,))
    xy = np.asarray(xy, dtype='float32')
    return xy


def get_landmarks(img, detector, predictor, PlotOn=False):
    # if not automatically downloaded, get it from:
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    # print(this_path)
    # predictor_path = this_path + "/dlib_models/shape_predictor_68_face_landmarks.dat"
    # print(predictor_path)
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(predictor_path)

    lmarks = []
    dets, scores, idx = detector.run(img, 1)
    # dets = [dlib.rectangle(left=0, top=0, right=img.shape[1], bottom=img.shape[0])]
    print("Number of faces detected: {}".format(len(dets)))
    if len(dets)>0:
        shapes = []
        for k, det in enumerate(dets):
            shape = predictor(img, det)
            shapes.append(shape)
            xy = _shape_to_np(shape)
            lmarks.append(xy)

        lmarks = np.asarray(lmarks, dtype='float32')
        lmarks = lmarks[0,:,:].T
        if PlotOn:
            display_landmarks(img, lmarks)
        return lmarks
    else:
        return lmarks


def display_landmarks(img, lmarks):
    for i in range(68):  
        xy = lmarks[:, i]  
        plt.plot(xy[0], xy[1], 'ro')  
        plt.text(xy[0], xy[1], str(i))
    plt.imshow(img)  
    plt.show()  