"""Normalize Bentham dataset."""

from glob import glob
import argparse
import sys
import os
import shutil

try:
    from settings.environment import Environment
except ImportError:
    sys.path[0] = os.path.join(sys.path[0], "..")
    from settings.environment import Environment


def norm_partitions(origin, env):
    """Normalize and create 'partitions' folder."""

    if os.path.exists(env.partitions_dir):
        shutil.rmtree(env.partitions_dir)
    os.makedirs(env.partitions_dir)

    origin_dir = os.path.join(origin, "BenthamDatasetR0-GT")

    set_file = os.path.join(origin_dir, "Partitions", "TrainLines.lst")
    shutil.copy(set_file, env.train_file)

    set_file = os.path.join(origin_dir, "Partitions", "ValidationLines.lst")
    shutil.copy(set_file, env.validation_file)

    set_file = os.path.join(origin_dir, "Partitions", "TestLines.lst")
    shutil.copy(set_file, env.test_file)


def norm_gt(origin, env):
    """Normalize and create 'gt' folder (Ground Truth)."""

    if os.path.exists(env.gt_dir):
        shutil.rmtree(env.gt_dir)
    os.makedirs(env.gt_dir)

    origin_dir = os.path.join(origin, "BenthamDatasetR0-GT")

    glob_filter = os.path.join(origin_dir, "Transcriptions", "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        shutil.copy(file, env.gt_dir)


def norm_data(origin, env):
    """Normalize and create 'lines' folder."""

    if os.path.exists(env.data_dir):
        shutil.rmtree(env.data_dir)
    os.makedirs(env.data_dir)

    origin_dir = os.path.join(origin, "BenthamDatasetR0-GT")

    glob_filter = os.path.join(origin_dir, "Images", "Lines", "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        name = os.path.basename(file).split(".")[0]
        new_file = os.path.join(env.data_dir, f"{name}.{env.extension}")
        shutil.copy(file, new_file)


def main():
    """Get the input parameter and call normalization methods."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()

    env = Environment(args.dataset_dir)
    src_backup = f"{args.dataset_dir}_backup"

    if not os.path.exists(src_backup):
        os.rename(args.dataset_dir, src_backup)

    norm_partitions(src_backup, env)
    norm_gt(src_backup, env)
    norm_data(src_backup, env)


if __name__ == '__main__':
    main()
