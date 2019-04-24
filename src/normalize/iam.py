"""Normalize IAM dataset"""

from glob import glob
import os
import shutil


def partitions(args):
    """Normalize and create 'partitions' folder"""

    if os.path.exists(args.PARTITIONS):
        shutil.rmtree(args.PARTITIONS)
    os.makedirs(args.PARTITIONS)

    origin_dir = os.path.join(args.SOURCE_BACKUP, "largeWriterIndependentTextLineRecognitionTask")
    set_file = os.path.join(origin_dir, "trainset.txt")
    shutil.copy(set_file, args.TRAIN_FILE)

    set_file1 = os.path.join(origin_dir, "validationset1.txt")
    set_file2 = os.path.join(origin_dir, "validationset2.txt")

    with open(args.VALIDATION_FILE, 'w') as outfile:
        with open(set_file1) as infile:
            outfile.write(infile.read())

        with open(set_file2) as infile:
            outfile.write(infile.read())

    set_file = os.path.join(origin_dir, "testset.txt")
    shutil.copy(set_file, args.TEST_FILE)


def ground_truth(args):
    """Normalize and create 'gt' folder (Ground Truth)"""

    if os.path.exists(args.GROUND_TRUTH):
        shutil.rmtree(args.GROUND_TRUTH)
    os.makedirs(args.GROUND_TRUTH)

    origin_dir = os.path.join(args.SOURCE_BACKUP, "ascii")
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

            new_set_file = os.path.join(args.GROUND_TRUTH, f"{file_name}.txt")

            with open(new_set_file, "w+") as new_file:
                new_file.write(file_text.strip())


def data(args):
    """Normalize and create 'lines' folder"""

    if os.path.exists(args.DATA):
        shutil.rmtree(args.DATA)
    os.makedirs(args.DATA)

    origin_dir = os.path.join(args.SOURCE_BACKUP, "lines")

    glob_filter = os.path.join(origin_dir, "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        name = os.path.basename(file).split(".")[0]
        new_file = os.path.join(args.DATA, f"{name}.png")
        shutil.copy(file, new_file)
