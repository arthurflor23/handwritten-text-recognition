"""Normalize Bentham dataset."""

from glob import glob
import os
import shutil


def partitions(origin, env):
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


def ground_truth(origin, env):
    """Normalize and create 'gt' folder (Ground Truth)."""

    if os.path.exists(env.gt_dir):
        shutil.rmtree(env.gt_dir)
    os.makedirs(env.gt_dir)

    origin_dir = os.path.join(origin, "BenthamDatasetR0-GT")

    glob_filter = os.path.join(origin_dir, "Transcriptions", "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        shutil.copy(file, env.gt_dir)


def data(origin, env):
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
