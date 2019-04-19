"""Normalize IAM dataset."""

from glob import glob
import argparse
import sys
import os
import shutil

try:
    from settings import Environment
except ImportError:
    sys.path[0] = os.path.join(sys.path[0], "..")
    from settings import Environment


def norm_partitions(origin, env):
    """Normalize and create 'partitions' folder."""

    if os.path.exists(env.partitions_dir):
        shutil.rmtree(env.partitions_dir)
    os.makedirs(env.partitions_dir)

    origin_dir = os.path.join(
        origin, "largeWriterIndependentTextLineRecognitionTask")

    set_file = os.path.join(origin_dir, "trainset.txt")
    shutil.copy(set_file, env.train_file)

    set_file1 = os.path.join(origin_dir, "validationset1.txt")
    set_file2 = os.path.join(origin_dir, "validationset2.txt")

    with open(env.validation_file, 'w') as outfile:
        with open(set_file1) as infile:
            outfile.write(infile.read())

        with open(set_file2) as infile:
            outfile.write(infile.read())

    set_file = os.path.join(origin_dir, "testset.txt")
    shutil.copy(set_file, env.test_file)


def norm_gt(origin, env):
    """Normalize and create 'gt' folder (Ground Truth)."""

    if os.path.exists(env.gt_dir):
        shutil.rmtree(env.gt_dir)
    os.makedirs(env.gt_dir)

    origin_dir = os.path.join(origin, "ascii")
    set_file = os.path.join(origin_dir, "lines.txt")

    with open(set_file) as file:
        content = [x.strip() for x in file.readlines()]

        for line in content:
            if (not line or line[0] == "#"):
                continue

            splited = line.strip().split(' ')
            assert len(splited) >= 9

            file_name = splited[0]
            file_text = splited[len(splited)-1].replace("|", " ")

            new_set_file = os.path.join(env.gt_dir, f"{file_name}.txt")

            with open(new_set_file, "w+") as new_file:
                new_file.write(file_text.strip())


def norm_data(origin, env):
    """Normalize and create 'lines' folder."""

    if os.path.exists(env.data_dir):
        shutil.rmtree(env.data_dir)
    os.makedirs(env.data_dir)

    origin_dir = os.path.join(origin, "lines")

    glob_filter = os.path.join(origin_dir, "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        name = os.path.basename(file).split(".")[0]
        new_file = os.path.join(env.data_dir, f"{name}.{env.extension}")
        shutil.copy(file, new_file)


def norm(args):
    """Get the input parameter and call normalization methods."""

    env = Environment(args.dataset_dir)
    src_backup = f"{args.dataset_dir}_backup"

    if not os.path.exists(src_backup):
        os.rename(args.dataset_dir, src_backup)

    norm_partitions(src_backup, env)
    norm_gt(src_backup, env)
    norm_data(src_backup, env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()
    norm(args)
