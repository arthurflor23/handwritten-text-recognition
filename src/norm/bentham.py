"""Normalize Bentham dataset."""

from glob import glob
import os
import shutil


def partitions(origin, path):
    """Normalize and create 'partitions' folder."""

    if os.path.exists(path.partitions):
        shutil.rmtree(path.partitions)
    os.makedirs(path.partitions)

    origin_dir = os.path.join(origin, "BenthamDatasetR0-GT")

    set_file = os.path.join(origin_dir, "Partitions", "TrainLines.lst")
    shutil.copy(set_file, path.train_file)

    set_file = os.path.join(origin_dir, "Partitions", "ValidationLines.lst")
    shutil.copy(set_file, path.validation_file)

    set_file = os.path.join(origin_dir, "Partitions", "TestLines.lst")
    shutil.copy(set_file, path.test_file)


def ground_truth(origin, path):
    """Normalize and create 'gt' folder (Ground Truth)."""

    if os.path.exists(path.ground_truth):
        shutil.rmtree(path.ground_truth)
    os.makedirs(path.ground_truth)

    origin_dir = os.path.join(origin, "BenthamDatasetR0-GT")

    glob_filter = os.path.join(origin_dir, "Transcriptions", "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        shutil.copy(file, path.ground_truth)


def data(origin, path):
    """Normalize and create 'lines' folder."""

    if os.path.exists(path.data):
        shutil.rmtree(path.data)
    os.makedirs(path.data)

    origin_dir = os.path.join(origin, "BenthamDatasetR0-GT")

    glob_filter = os.path.join(origin_dir, "Images", "Lines", "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        name = os.path.basename(file).split(".")[0]
        new_file = os.path.join(path.data, f"{name}.png")
        shutil.copy(file, new_file)
