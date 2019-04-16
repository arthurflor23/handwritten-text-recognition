"""Normalize Bentham dataset."""

from glob import glob
import argparse
import os
import shutil


def norm_partitions(origin, target):
    """Normalize and create 'partitions' folder."""

    origin_dir = os.path.join(origin, "BenthamDatasetR0-GT")
    target_dir = os.path.join(target, "partitions")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    set_file = os.path.join(origin_dir, "Partitions", "TrainLines.lst")
    new_set_file = os.path.join(target_dir, "train.txt")
    shutil.copy(set_file, new_set_file)

    set_file = os.path.join(origin_dir, "Partitions", "ValidationLines.lst")
    new_set_file = os.path.join(target_dir, "validation.txt")
    shutil.copy(set_file, new_set_file)

    set_file = os.path.join(origin_dir, "Partitions", "TestLines.lst")
    new_set_file = os.path.join(target_dir, "test.txt")
    shutil.copy(set_file, new_set_file)


def norm_gt(origin, target):
    """Normalize and create 'gt' folder (Ground Truth)."""

    origin_dir = os.path.join(origin, "BenthamDatasetR0-GT")
    target_dir = os.path.join(target, "gt")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    glob_filter = os.path.join(origin_dir, "Transcriptions", "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        shutil.copy(file, target_dir)


def norm_lines(origin, target):
    """Normalize and create 'lines' folder."""

    origin_dir = os.path.join(origin, "BenthamDatasetR0-GT")
    target_dir = os.path.join(target, "lines")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    glob_filter = os.path.join(origin_dir, "Images", "Lines", "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        shutil.copy(file, target_dir)


def main():
    """Get the input parameter and call normalization methods."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    src = args.data_dir
    src_backup = f"{src}_backup"

    if not os.path.exists(src_backup):
        os.rename(src, src_backup)

    norm_partitions(src_backup, src)
    norm_gt(src_backup, src)
    norm_lines(src_backup, src)


if __name__ == '__main__':
    main()
