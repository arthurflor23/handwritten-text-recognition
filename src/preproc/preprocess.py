"""Preprocess image to the model"""

import numpy as np
import cv2
from . import deslant


def preprocess(img, input_shape):
    """Image: remove cursive style, resize and normalization"""

    model_h, model_w = input_shape
    fac = max((img.shape[1] / model_w), (img.shape[0] / model_h))

    new_size = (max(min(model_w, int(img.shape[1] / fac)), 1),
                max(min(model_h, int(img.shape[0] / fac)), 1))

    img = deslant.remove_cursive_style(img)
    img = cv2.resize(img, new_size)

    target = np.ones([model_h, model_w]) * 255
    target[0:img.shape[0], 0:img.shape[1]] = img
    img = cv2.transpose(target)

    mean, stddev = cv2.meanStdDev(img)
    img = (img - mean[0][0])
    img = (img / stddev[0][0]) if stddev[0][0] > 0 else img

    return img
