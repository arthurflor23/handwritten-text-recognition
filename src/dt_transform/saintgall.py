"""Transform Saint Gall dataset"""

from multiprocessing import Pool
from functools import partial
from glob import glob
import numpy as np
import os


def dataset(env, preproc_func):
    """Load and save npz file of the ground truth and images (preprocessed)"""

    gt = os.path.join(env.raw_source, "ground_truth")
    ground_truth = open(os.path.join(gt, "transcription.txt")).read().splitlines()
    gt_dict = dict()

    for line in ground_truth:
        if (not line or line[0] == "#"):
            continue

        splited = line.strip().split(" ")
        assert len(splited) >= 3

        name = splited[0].strip()
        text = splited[1].replace("-", "").replace("|", " ").strip()
        gt_dict[name] = text

    dt, gt = build_data_from(env, "train.txt", gt_dict, preproc_func)
    np.savez_compressed(env.train, dt=dt, gt=gt)

    dt, gt = build_data_from(env, "valid.txt", gt_dict, preproc_func)
    np.savez_compressed(env.valid, dt=dt, gt=gt)

    dt, gt = build_data_from(env, "test.txt", gt_dict, preproc_func)
    np.savez_compressed(env.test, dt=dt, gt=gt)


def build_data_from(env, partition, gt_dict, preproc_func):
    """Preprocess images with pool function"""

    pt_path = os.path.join(env.raw_source, "sets")
    lines = open(os.path.join(pt_path, partition)).read().splitlines()
    data_path = os.path.join(env.raw_source, "data", "line_images_normalized")
    dt, gt = [], []

    for line in lines:
        glob_filter = os.path.join(data_path, f"{line}*")
        img_list = [x for x in glob(glob_filter, recursive=True)]

        for path in img_list:
            index = os.path.splitext(os.path.basename(path))[0]
            gt.append(gt_dict[index])
            dt.append(path)

    pool = Pool()
    dt = pool.map(partial(preproc_func, img_size=env.input_img_size, read_first=True), dt)
    pool.close()
    pool.join()

    return dt, gt
