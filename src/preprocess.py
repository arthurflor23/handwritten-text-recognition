"""Preprocess 'lines' folder of the dataset."""

import argparse
from data import Data


def preproc(data):
    """Preprocessor the 'lines' folder."""

    print(data.dataset)


def main():
    """Get the input parameter and call preproc method."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    data = Data(args.input_dir)
    data.imread_partitions()
    # preproc(data)


if __name__ == '__main__':
    main()
