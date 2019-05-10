"""Transform Saint Gall dataset"""

from multiprocessing import Pool
from functools import partial
from glob import glob
import numpy as np
import os


def dataset(env, preproc, encode):
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

    train_dt, train_gt = build_data(env, "train.txt", gt_dict, preproc, encode)
    valid_dt, valid_gt = build_data(env, "valid.txt", gt_dict, preproc, encode)
    test_dt, test_gt = build_data(env, "test.txt", gt_dict, preproc, encode)

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

    pt_path = os.path.join(env.raw_source, "sets")
    lines = open(os.path.join(pt_path, partition)).read().splitlines()
    data_path = os.path.join(env.raw_source, "data", "line_images_normalized")
    dt, gt = [], []

    for line in lines:
        glob_filter = os.path.join(data_path, f"{line}*")
        img_list = [x for x in glob(glob_filter, recursive=True)]

        for path in img_list:
            index = os.path.splitext(os.path.basename(path))[0]
            text_line = gt_dict[index].strip()

            if len(text_line) > 0:
                gt.append(text_line)
                dt.append(path)

    pool = Pool()
    dt = pool.map(partial(preproc, img_size=env.model_input_size, read_first=True), dt)
    gt = pool.map(partial(encode, charset=env.charset), gt)
    pool.close()
    pool.join()

    return dt, gt
