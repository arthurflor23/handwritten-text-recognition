from data import Data
import argparse


def preprocess(input):
    print(input.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    dt = Data(args.input_dir)
    # dt.load_dataset()
    preprocess(dt)


if __name__ == '__main__':
    main()
