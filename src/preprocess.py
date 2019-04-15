from data import Data
import argparse


def preprocess(data):
    print(data.train)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--subsets_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    dt = Data(args)
    dt.load_dataset()
    preprocess(dt)


if __name__ == '__main__':
    main()
