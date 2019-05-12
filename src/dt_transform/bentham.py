"""Transform Bentham dataset"""

from multiprocessing import Pool
from functools import partial
import h5py
import os


def dataset(env, preproc, encode):
    """Load and save hdf5 file of the ground truth and images (preprocessed)"""

    def transform(group, target):
        with h5py.File(env.source, "a") as hf:
            dt, gt = build_data(env, target, gt_dict, preproc, encode)
            hf.create_dataset(f"{group}/dt", data=dt, compression="gzip", compression_opts=9)
            hf.create_dataset(f"{group}/gt", data=gt, compression="gzip", compression_opts=9)
            del dt, gt

    env.raw_source = os.path.join(env.raw_source, "BenthamDatasetR0-GT")
    path = os.path.join(env.raw_source, "Transcriptions")

    gt = os.listdir(path=path)
    gt_dict = dict()

    for x in gt:
        text = " ".join(open(os.path.join(path, x)).read().splitlines()).replace("_", "")
        gt_dict[os.path.splitext(x)[0]] = text.strip()

    transform(group="train", target="TrainLines.lst")
    transform(group="valid", target="ValidationLines.lst")
    transform(group="test", target="TestLines.lst")


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
            dt.append(path)
            gt.append(text_line)

    pool = Pool()
    dt = pool.map(partial(preproc, img_size=env.input_size, read_first=True), dt)
    gt = pool.map(partial(encode, charset=env.charset, mtl=env.max_text_length), gt)
    pool.close()
    pool.join()

    return dt, gt
