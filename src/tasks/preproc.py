"""Preprocessor 'lines' folder of the dataset"""

from multiprocessing import Pool
from functools import partial
import sys
import os
import argparse
import shutil

try:
    sys.path[0] = os.path.join(sys.path[0], "..")
    from environment import setup_path
    from preproc.preprocess import preprocess
    from network import model
except ImportError as exc:
    sys.exit(f"Import error in '{__file__}': {exc}")


def main():
    """Preprocess data folder of the dataset"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../output")
    args = parser.parse_args()

    args = setup_path(args)
    data_list = []

    with open(args.TRAIN_FILE, "r") as file:
        data_list += [x.strip() for x in file.readlines()]

    with open(args.VALIDATION_FILE, "r") as file:
        data_list += [x.strip() for x in file.readlines()]

    with open(args.TEST_FILE, "r") as file:
        data_list += [x.strip() for x in file.readlines()]

    if os.path.exists(args.PREPROC):
        shutil.rmtree(args.PREPROC)
    os.makedirs(args.PREPROC)

    pool = Pool()
    pool.map(partial(preprocess, args=args, img_size=model.INPUT_SIZE), data_list)
    pool.close()


if __name__ == '__main__':
    main()
