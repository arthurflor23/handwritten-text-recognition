"""Transform Bentham dataset"""

from multiprocessing import Pool
from functools import partial
import numpy as np
import os


def dataset(env, preproc_func):
    """Load and save npz file of the ground truth and images (preprocessed)"""

    env.raw_source = os.path.join(env.raw_source, "BenthamDatasetR0-GT")
    path = os.path.join(env.raw_source, "Transcriptions")

    gt = os.listdir(path=path)
    gt_dict = dict()

    for x in gt:
        text = " ".join(open(os.path.join(path, x)).read().splitlines()).replace("_", "")
        gt_dict[os.path.splitext(x)[0]] = text.strip()

    dt, gt = build_data_from(env, "TrainLines.lst", gt_dict, preproc_func)
    np.savez_compressed(env.train, dt=dt, gt=gt)

    dt, gt = build_data_from(env, "ValidationLines.lst", gt_dict, preproc_func)
    np.savez_compressed(env.valid, dt=dt, gt=gt)

    dt, gt = build_data_from(env, "TestLines.lst", gt_dict, preproc_func)
    np.savez_compressed(env.test, dt=dt, gt=gt)


def build_data_from(env, partition, gt_dict, preproc_func):
    """Preprocess images with pool function"""

    pt_path = os.path.join(env.raw_source, "Partitions")
    lines = open(os.path.join(pt_path, partition)).read().splitlines()
    data_path = os.path.join(env.raw_source, "Images", "Lines")
    dt, gt = [], []

    for line in lines:
        path = os.path.join(data_path, f"{line}.png")
        gt.append(gt_dict[line])
        dt.append(path)

    pool = Pool()
    dt = pool.map(partial(preproc_func, img_size=env.model_input_size, read_first=True), dt)
    pool.close()
    pool.join()

    return dt, gt
