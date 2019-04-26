"""Transform Saint Gall dataset"""

from glob import glob
import shutil
import os


def partitions(args):
    """Transform and create 'partitions' folder"""

    if os.path.exists(args.PARTITIONS):
        shutil.rmtree(args.PARTITIONS)
    os.makedirs(args.PARTITIONS)

    origin_dir = os.path.join(args.RAW_SOURCE, "sets")

    def complete_partition_file(set_file, new_set_file):
        lines = os.path.join(args.SOURCE, "data", "line_images_normalized")

        with open(set_file) as file:
            with open(new_set_file, "w+") as new_file:
                content = [x.strip() for x in file.readlines()]

                for item in content:
                    glob_filter = os.path.join(lines, f"{item}*")
                    paths = [x for x in glob(glob_filter, recursive=True)]

                    for path in paths:
                        basename = os.path.basename(path).split(".")[0]
                        new_file.write(f"{basename.strip()}\n")

    set_file = os.path.join(origin_dir, "train.txt")
    complete_partition_file(set_file, args.TRAIN_FILE)

    set_file = os.path.join(origin_dir, "valid.txt")
    complete_partition_file(set_file, args.VALIDATION_FILE)

    set_file = os.path.join(origin_dir, "test.txt")
    complete_partition_file(set_file, args.TEST_FILE)


def ground_truth(args):
    """Transform and create 'ground_truth' folder (Ground Truth)"""

    if os.path.exists(args.GROUND_TRUTH):
        shutil.rmtree(args.GROUND_TRUTH)
    os.makedirs(args.GROUND_TRUTH)

    origin_dir = os.path.join(args.RAW_SOURCE, "ground_truth")
    set_file = os.path.join(origin_dir, "transcription.txt")

    with open(set_file) as file:
        content = [x.strip() for x in file.readlines()]

        for line in content:
            if (not line or line[0] == "#"):
                continue

            splited = line.strip().split(' ')
            assert len(splited) >= 3

            file_name = splited[0]
            file_text = splited[1].replace("-", "").replace("|", " ")
            new_set_file = os.path.join(args.GROUND_TRUTH, f"{file_name}.txt")

            with open(new_set_file, "w+") as new_file:
                new_file.write(file_text.strip())


def data(args):
    """Transform and create 'lines' folder"""

    if os.path.exists(args.DATA):
        shutil.rmtree(args.DATA)
    os.makedirs(args.DATA)

    origin_dir = os.path.join(args.RAW_SOURCE, "data")
    glob_filter = os.path.join(origin_dir, "line_images_normalized", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        name = os.path.basename(file).split(".")[0]
        new_file = os.path.join(args.DATA, f"{name}.png")
        shutil.copy(file, new_file)
