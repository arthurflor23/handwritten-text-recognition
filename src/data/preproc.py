"""Preprocessor 'lines' folder of the dataset."""

import argparse
try:
    from .loader import Loader
except ImportError:
    from loader import Loader


def preprocess(data):
    """Preprocess 'lines' folder."""

    print(data.dataset)


def main():
    """Get the input parameter and call preprocess method."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    data = Loader(args.input_dir)
    data.imread_partitions()
    # preprocess(data)


if __name__ == '__main__':
    main()
