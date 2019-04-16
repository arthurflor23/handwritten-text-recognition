from glob import glob
import argparse
import shutil
import os


def norm_partitions(origin, target):
    return


def norm_gt(origin, target):
    return


def norm_lines(origin, target):
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    src = os.path.join(args.input_dir, os.path.basename(__file__)[:-3])
    src_backup = f"{src}_backup"

    if not os.path.exists(src_backup):
        os.rename(src, src_backup)

    norm_partitions(src_backup, src)
    # norm_gt(src_backup, src)
    # norm_lines(src_backup, src)


if __name__ == '__main__':
    main()
