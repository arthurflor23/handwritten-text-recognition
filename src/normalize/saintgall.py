from glob import glob
import argparse
import shutil
import os


def norm_partitions(origin, target):
    origin_dir = os.path.join(origin, "sets")
    target_dir = os.path.join(target, "partitions")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    def complete_partition_file(set_file, new_set_file):
        with open(set_file) as f:
            with open(new_set_file, "w+") as new_file:
                content = [x.strip() for x in f.readlines()]
                lines = os.path.join(origin, "data", "line_images_normalized")

                for item in content:
                    glob_filter = os.path.join(lines, f"{item}*")
                    paths = [x for x in glob(glob_filter, recursive=True)]

                    for path in paths:
                        basename = os.path.basename(path).split(".")[0]
                        new_file.write(f"{basename.strip()}\n")

                new_file.close()

    set_file = os.path.join(origin_dir, "train.txt")
    new_set_file = os.path.join(target_dir, "train.txt")
    complete_partition_file(set_file, new_set_file)

    set_file = os.path.join(origin_dir, "valid.txt")
    new_set_file = os.path.join(target_dir, "validation.txt")
    complete_partition_file(set_file, new_set_file)

    set_file = os.path.join(origin_dir, "test.txt")
    new_set_file = os.path.join(target_dir, "test.txt")
    complete_partition_file(set_file, new_set_file)


def norm_gt(origin, target):
    origin_dir = os.path.join(origin, "ground_truth")
    target_dir = os.path.join(target, "gt")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    set_file = os.path.join(origin_dir, "transcription.txt")

    with open(set_file) as f:
        content = [x.strip() for x in f.readlines()]

        for line in content:
            if (not line or line[0] == "#"):
                continue

            splited = line.strip().split(' ')
            assert len(splited) >= 3

            file_name = splited[0]
            file_text = splited[1].replace("-", "").replace("|", " ")

            new_set_file = os.path.join(target_dir, f"{file_name}.txt")

            with open(new_set_file, "w+") as f:
                f.write(file_text.strip())
                f.close()


def norm_lines(origin, target):
    origin_dir = os.path.join(origin, "data")
    target_dir = os.path.join(target, "lines")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    glob_filter = os.path.join(origin_dir, "line_images_normalized", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for f in files:
        shutil.copy(f, target_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    src = os.path.join(args.input_dir, os.path.basename(__file__)[:-3])
    src_backup = f"{src}_backup"

    if not os.path.exists(src_backup):
        os.rename(src, src_backup)

    norm_partitions(src_backup, src)
    norm_gt(src_backup, src)
    norm_lines(src_backup, src)


if __name__ == '__main__':
    main()
