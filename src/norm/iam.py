"""Normalize IAM dataset"""

from glob import glob
import os
import shutil


def partitions(origin, path):
    """Normalize and create 'partitions' folder"""

    if os.path.exists(path.partitions):
        shutil.rmtree(path.partitions)
    os.makedirs(path.partitions)

    origin_dir = os.path.join(origin, "largeWriterIndependentTextLineRecognitionTask")
    set_file = os.path.join(origin_dir, "trainset.txt")
    shutil.copy(set_file, path.train_file)

    set_file1 = os.path.join(origin_dir, "validationset1.txt")
    set_file2 = os.path.join(origin_dir, "validationset2.txt")

    with open(path.validation_file, 'w') as outfile:
        with open(set_file1) as infile:
            outfile.write(infile.read())

        with open(set_file2) as infile:
            outfile.write(infile.read())

    set_file = os.path.join(origin_dir, "testset.txt")
    shutil.copy(set_file, path.test_file)


def ground_truth(origin, path):
    """Normalize and create 'gt' folder (Ground Truth)"""

    if os.path.exists(path.ground_truth):
        shutil.rmtree(path.ground_truth)
    os.makedirs(path.ground_truth)

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
            file_text = splited[len(splited) - 1].replace("|", " ")

            new_set_file = os.path.join(path.ground_truth, f"{file_name}.txt")

            with open(new_set_file, "w+") as new_file:
                new_file.write(file_text.strip())


def data(origin, path):
    """Normalize and create 'lines' folder"""

    if os.path.exists(path.data):
        shutil.rmtree(path.data)
    os.makedirs(path.data)

    origin_dir = os.path.join(origin, "lines")

    glob_filter = os.path.join(origin_dir, "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        name = os.path.basename(file).split(".")[0]
        new_file = os.path.join(path.data, f"{name}.png")
        shutil.copy(file, new_file)
