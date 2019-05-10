"""Transform Bentham dataset"""

from multiprocessing import Pool
from functools import partial
import numpy as np
import os


def dataset(env, preproc, encode):
    """Load and save npz file of the ground truth and images (preprocessed)"""

    env.raw_source = os.path.join(env.raw_source, "BenthamDatasetR0-GT")
    path = os.path.join(env.raw_source, "Transcriptions")

    gt = os.listdir(path=path)
    gt_dict = dict()

    for x in gt:
        text = " ".join(open(os.path.join(path, x)).read().splitlines()).replace("_", "")
        gt_dict[os.path.splitext(x)[0]] = text.strip()

    train_dt, train_gt = build_data(env, "TrainLines.lst", gt_dict, preproc, encode)
    valid_dt, valid_gt = build_data(env, "ValidationLines.lst", gt_dict, preproc, encode)
    test_dt, test_gt = build_data(env, "TestLines.lst", gt_dict, preproc, encode)

    np.savez_compressed(
        env.source,
        train_dt=train_dt,
        train_gt=train_gt,
        valid_dt=valid_dt,
        valid_gt=valid_gt,
        test_dt=test_dt,
        test_gt=test_gt,
    )


def build_data(env, partition, gt_dict, preproc, encode):
    """Preprocess images with pool function"""

    pt_path = os.path.join(env.raw_source, "Partitions")
    lines = open(os.path.join(pt_path, partition)).read().splitlines()
    data_path = os.path.join(env.raw_source, "Images", "Lines")
    dt, gt = [], []

    for line in lines:
        text_line = gt_dict[line].strip()

        if len(text_line) > 0:
            path = os.path.join(data_path, f"{line}.png")
            gt.append(text_line)
            dt.append(path)

    pool = Pool()
    dt = pool.map(partial(preproc, img_size=env.model_input_size, read_first=True), dt)
    gt = pool.map(partial(encode, charset=env.charset), gt)
    pool.close()
    pool.join()

    return dt, gt
