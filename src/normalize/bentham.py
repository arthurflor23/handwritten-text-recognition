"""Normalize Bentham dataset"""

from glob import glob
import os
import shutil


def partitions(args):
    """Normalize and create 'partitions' folder"""

    if os.path.exists(args.PARTITIONS):
        shutil.rmtree(args.PARTITIONS)
    os.makedirs(args.PARTITIONS)

    origin_dir = os.path.join(args.SOURCE_BACKUP, "BenthamDatasetR0-GT")

    set_file = os.path.join(origin_dir, "Partitions", "TrainLines.lst")
    shutil.copy(set_file, args.TRAIN_FILE)

    set_file = os.path.join(origin_dir, "Partitions", "ValidationLines.lst")
    shutil.copy(set_file, args.VALIDATION_FILE)

    set_file = os.path.join(origin_dir, "Partitions", "TestLines.lst")
    shutil.copy(set_file, args.TEST_FILE)


def ground_truth(args):
    """Normalize and create 'gt' folder (Ground Truth)"""

    if os.path.exists(args.GROUND_TRUTH):
        shutil.rmtree(args.GROUND_TRUTH)
    os.makedirs(args.GROUND_TRUTH)

    origin_dir = os.path.join(args.SOURCE_BACKUP, "BenthamDatasetR0-GT")

    glob_filter = os.path.join(origin_dir, "Transcriptions", "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        shutil.copy(file, args.GROUND_TRUTH)


def data(args):
    """Normalize and create 'lines' folder"""

    if os.path.exists(args.DATA):
        shutil.rmtree(args.DATA)
    os.makedirs(args.DATA)

    origin_dir = os.path.join(args.SOURCE_BACKUP, "BenthamDatasetR0-GT")

    glob_filter = os.path.join(origin_dir, "Images", "Lines", "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        name = os.path.basename(file).split(".")[0]
        new_file = os.path.join(args.DATA, f"{name}.png")
        shutil.copy(file, new_file)
