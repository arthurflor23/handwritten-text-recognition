"""Methods of image binarization"""

import os
import numpy as np
import cv2
from . import binarization, deslant


def preprocess(filename, args, img_size):
    """Read, preprocess and save new image"""

    img_path = os.path.join(args.DATA, f"{filename}.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        img = np.zeros(img_size[1::-1], dtype=np.uint8)

    model_w, model_h = img_size[:2]
    fac = max((img.shape[1] / model_w), (img.shape[0] / model_h))

    new_size = (max(min(model_w, int(img.shape[1] / fac)), 1),
                max(min(model_h, int(img.shape[0] / fac)), 1))

    ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = binarization.sauvola(img, [25, 25], 127, 0.02) if ret > 127 else otsu

    img = deslant.remove_cursive_style(img)
    img = cv2.resize(img, new_size)

    target = np.ones([model_h, model_w]) * 255
    target[0:new_size[1], 0:new_size[0]] = img
    img = cv2.transpose(target)

    mean, stddev = cv2.meanStdDev(img)
    img = (img - mean[0][0])
    img = (img / stddev[0][0]) if stddev[0][0] > 0 else img

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    np.save(os.path.join(args.PREPROC, filename), img)
