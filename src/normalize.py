"""Dataset normalizer"""

import argparse
import importlib
import os

from settings import environment as env


def main():
    """Get the input parameter and call normalization method."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()

    package = f"norm.{os.path.basename(args.dataset_dir)}"
    normalize = importlib.import_module(package)

    path = env.Path(args.dataset_dir)
    src_backup = f"{args.dataset_dir}_backup"

    if not os.path.exists(src_backup):
        os.rename(args.dataset_dir, src_backup)

    normalize.partitions(src_backup, path)
    normalize.ground_truth(src_backup, path)
    normalize.data(src_backup, path)


if __name__ == '__main__':
    main()
