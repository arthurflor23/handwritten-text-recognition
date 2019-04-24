"""Preprocessor 'lines' folder of the dataset."""

from multiprocessing import Pool
from functools import partial
from preproc import deslant, binarization
from settings import Environment, INPUT_SIZE

import argparse
import shutil
import numpy as np
import cv2
import os


def imread(env):
    """Load image list names from partitions texts"""

    def imread_partition(txt):
        with open(txt, "r") as file:
            return [x.strip() for x in file.readlines()]

    data_list = []
    data_list += imread_partition(env.train_file)
    data_list += imread_partition(env.validation_file)
    data_list += imread_partition(env.test_file)
    return data_list


def preprocess(filename, env):
    """Read, preprocess and save new image."""

    img_path = os.path.join(env.data_dir, f"{filename}.{env.extension}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        img = np.zeros(INPUT_SIZE[1::-1], dtype=np.uint8)

    env_w, env_h = INPUT_SIZE[:2]
    img_h, img_w = img.shape
    fac = max((img_w/env_w), (img_h/env_h))

    new_size = (max(min(env_w, int(img_w/fac)), 1),
                max(min(env_h, int(img_h/fac)), 1))

    ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = binarization.sauvola(img, [25, 25], 127, 0.02) if ret > 127 else otsu

    img = deslant.remove_cursive_style(img)
    img = cv2.resize(img, new_size)

    target = np.ones([env_h, env_w]) * 255
    target[0:new_size[1], 0:new_size[0]] = img
    img = cv2.transpose(target)

    mean, stddev = cv2.meanStdDev(img)
    img = (img-mean[0][0])
    img = (img/stddev[0][0]) if stddev[0][0] > 0 else img

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    np.save(os.path.join(env.preproc_dir, filename), img)


def main():
    """Preprocess data folder of the dataset."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()

    env = Environment(args.dataset_dir)
    data_list = imread(env)

    if os.path.exists(env.preproc_dir):
        shutil.rmtree(env.preproc_dir)
    os.makedirs(env.preproc_dir)

    pool = Pool()
    pool.map(partial(preprocess, env=env), data_list)
    pool.close()


if __name__ == '__main__':
    main()
