"""Dataset normalizer"""

import sys
import os
import argparse
import importlib

try:
    sys.path[0] = os.path.join(sys.path[0], "..")
    from environment import setup_path
except ImportError as exc:
    sys.exit(f"Import error in '{__file__}': {exc}")


def main():
    """Get the input parameter and call normalization method"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../output")
    args = parser.parse_args()

    args = setup_path(args)

    package = f"normalize.{os.path.basename(args.SOURCE)}"
    normalize = importlib.import_module(package)

    if not os.path.exists(args.SOURCE_BACKUP):
        os.rename(args.SOURCE, args.SOURCE_BACKUP)

    normalize.partitions(args)
    normalize.ground_truth(args)
    normalize.data(args)


if __name__ == '__main__':
    main()
