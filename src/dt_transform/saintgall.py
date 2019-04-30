"""Transform Saint Gall dataset"""

from glob import glob
import shutil
import os


def partitions(args):
    """Transform and create 'partitions' folder"""

    if os.path.exists(args.partitions):
        shutil.rmtree(args.partitions)
    os.makedirs(args.partitions)

    origin_dir = os.path.join(args.raw_source, "sets")

    def complete_partition_file(set_file, new_set_file):
        lines = os.path.join(args.source, "data", "line_images_normalized")

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
    complete_partition_file(set_file, args.train_file)

    set_file = os.path.join(origin_dir, "valid.txt")
    complete_partition_file(set_file, args.validation_file)

    set_file = os.path.join(origin_dir, "test.txt")
    complete_partition_file(set_file, args.test_file)


def ground_truth(args):
    """Transform and create 'ground_truth' folder (Ground Truth)"""

    if os.path.exists(args.ground_truth):
        shutil.rmtree(args.ground_truth)
    os.makedirs(args.ground_truth)

    origin_dir = os.path.join(args.raw_source, "ground_truth")
    set_file = os.path.join(origin_dir, "transcription.txt")

    with open(set_file) as file:
        content = [x.strip() for x in file.readlines()]

        for line in content:
            if (not line or line[0] == "#"):
                continue

            splited = line.strip().split(" ")
            assert len(splited) >= 3

            file_name = splited[0]
            file_text = splited[1].replace("-", "").replace("|", " ")
            new_set_file = os.path.join(args.ground_truth, f"{file_name}.txt")

            with open(new_set_file, "w+") as new_file:
                new_file.write(file_text.strip())


def data(args):
    """Transform and create 'lines' folder"""

    if os.path.exists(args.data):
        shutil.rmtree(args.data)
    os.makedirs(args.data)

    origin_dir = os.path.join(args.raw_source, "data")
    glob_filter = os.path.join(origin_dir, "line_images_normalized", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for file in files:
        name = os.path.basename(file).split(".")[0]
        new_file = os.path.join(args.data, f"{name}.png")
        shutil.copy(file, new_file)
