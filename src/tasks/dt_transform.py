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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, required=True)
    parser.add_argument("--data_output", type=str, default="../output")
    args = parser.parse_args()

    args = setup_path(args)

    package = f"dt_transform.{os.path.basename(args.SOURCE)}"
    transform = importlib.import_module(package)

    if not os.path.exists(args.RAW_SOURCE):
        os.rename(args.SOURCE, args.RAW_SOURCE)

    transform.partitions(args)
    transform.ground_truth(args)
    transform.data(args)


if __name__ == '__main__':
    main()
