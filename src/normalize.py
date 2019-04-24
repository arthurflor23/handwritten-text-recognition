"""Dataset normalizer"""

from settings import Environment

import argparse
import importlib
import os


def main():
    """Get the input parameter and call normalization method."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()

    package = f"norm.{os.path.basename(args.dataset_dir)}"
    normalize = importlib.import_module(package)

    env = Environment(args.dataset_dir)
    src_backup = f"{args.dataset_dir}_backup"

    if not os.path.exists(src_backup):
        os.rename(args.dataset_dir, src_backup)

    normalize.partitions(src_backup, env)
    normalize.ground_truth(src_backup, env)
    normalize.data(src_backup, env)


if __name__ == '__main__':
    main()
