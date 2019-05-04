"""Transform IAM dataset"""

from multiprocessing import Pool
from functools import partial
import numpy as np
import os


def dataset(env, preproc_func):
    """Load and save npz file of the ground truth and images (preprocessed)"""

    gt = os.path.join(env.raw_source, "ascii")
    ground_truth = open(os.path.join(gt, "lines.txt")).read().splitlines()
    gt_dict = dict()

    for line in ground_truth:
        if (not line or line[0] == "#"):
            continue

        splited = line.strip().split(" ")
        assert len(splited) >= 9

        name = splited[0].strip()
        text = splited[len(splited) - 1].replace("|", " ").strip()
        gt_dict[name] = text

    dt, gt = build_data_from(env, "trainset.txt", gt_dict, preproc_func)
    np.savez_compressed(env.train, dt=dt, gt=gt)

    dt, gt = build_data_from(env, "validationset2.txt", gt_dict, preproc_func)
    np.savez_compressed(env.valid, dt=dt, gt=gt)

    dt, gt = build_data_from(env, "testset.txt", gt_dict, preproc_func)
    np.savez_compressed(env.test, dt=dt, gt=gt)


def build_data_from(env, partition, gt_dict, preproc_func):
    """Preprocess images with pool function"""

    pt_path = os.path.join(env.raw_source, "largeWriterIndependentTextLineRecognitionTask")
    lines = open(os.path.join(pt_path, partition)).read().splitlines()
    data_path = os.path.join(env.raw_source, "lines")
    dt, gt = [], []

    for line in lines:
        text_line = gt_dict[line].strip()

        if len(text_line) > 0:
            split = line.split("-")
            path = os.path.join(split[0], f"{split[0]}-{split[1]}", f"{split[0]}-{split[1]}-{split[2]}.png")
            path = os.path.join(data_path, path)

            gt.append(text_line)
            dt.append(path)

    pool = Pool()
    dt = pool.map(partial(preproc_func, img_size=env.model_input_size, read_first=True), dt)
    pool.close()
    pool.join()

    return dt, gt
