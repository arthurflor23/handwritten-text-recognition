"""Transform Bentham dataset"""

from glob import glob
import os
import shutil


def partitions(args):
    """Transform and create 'partitions' folder"""

    if os.path.exists(args.partitions):
        shutil.rmtree(args.partitions)
    os.makedirs(args.partitions)

    origin_dir = os.path.join(args.raw_source, "BenthamDatasetR0-GT")

    set_file = os.path.join(origin_dir, "Partitions", "TrainLines.lst")
    shutil.copy(set_file, args.train_file)

    set_file = os.path.join(origin_dir, "Partitions", "ValidationLines.lst")
    shutil.copy(set_file, args.validation_file)

    set_file = os.path.join(origin_dir, "Partitions", "TestLines.lst")
    shutil.copy(set_file, args.test_file)


def ground_truth(args):
    """Transform and create 'gt' folder (Ground Truth)"""

    if os.path.exists(args.ground_truth):
        shutil.rmtree(args.ground_truth)
    os.makedirs(args.ground_truth)

    origin_dir = os.path.join(args.raw_source, "BenthamDatasetR0-GT")

    glob_filter = os.path.join(origin_dir, "Transcriptions", "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        shutil.copy(file, args.ground_truth)


def data(args):
    """Transform and create 'lines' folder"""

    if os.path.exists(args.data):
        shutil.rmtree(args.data)
    os.makedirs(args.data)

    origin_dir = os.path.join(args.raw_source, "BenthamDatasetR0-GT")

    glob_filter = os.path.join(origin_dir, "Images", "Lines", "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        name = os.path.basename(file).split(".")[0]
        new_file = os.path.join(args.data, f"{name}.png")
        shutil.copy(file, new_file)
