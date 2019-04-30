"""Transform dataset to the project standard"""

import sys
import os
import argparse
import importlib

try:
    sys.path[0] = os.path.join(sys.path[0], "..")
    from environment import setup_path
except ImportError as exc:
    sys.exit(f"Import error in '{__file__}': {exc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    args = setup_path(args)

    package = f"dt_transform.{os.path.basename(args.source)}"
    transform = importlib.import_module(package)

    if not os.path.exists(args.raw_source):
        os.rename(args.source, args.raw_source)

    transform.partitions(args)
    transform.ground_truth(args)
    transform.data(args)
