"""Preprocessor 'lines' folder of the dataset"""

from multiprocessing import Pool
from functools import partial
import argparse
import os
import shutil
import numpy as np
import cv2

from preproc import deslant, binarization
from settings import model, environment as env


def imread(train_file, validation_file, test_file):
    """Load image list names from partitions texts"""

    def imread_partition(txt):
        with open(txt, "r") as file:
            return [x.strip() for x in file.readlines()]

    data_list = []
    data_list += imread_partition(train_file)
    data_list += imread_partition(validation_file)
    data_list += imread_partition(test_file)
    return data_list


def preprocess(filename, data_dir, preproc_dir):
    """Read, preprocess and save new image"""

    img_path = os.path.join(data_dir, f"{filename}.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        img = np.zeros(model.INPUT_SIZE[1::-1], dtype=np.uint8)

    model_w, model_h = model.INPUT_SIZE[:2]
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
    np.save(os.path.join(preproc_dir, filename), img)


def main():
    """Preprocess data folder of the dataset"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()

    path = env.Path(args.dataset_dir)
    data_list = imread(path.train_file, path.validation_file, path.test_file)

    if os.path.exists(path.preproc):
        shutil.rmtree(path.preproc)
    os.makedirs(path.preproc)

    pool = Pool()
    pool.map(partial(preprocess, data_dir=path.data, preproc_dir=path.preproc), data_list)
    pool.close()


if __name__ == '__main__':
    main()
