from data import Data
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, default=None, required=True)
    parser.add_argument("--dataset", type=str, default=None, required=True)

    parser.add_argument("--preprocess", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)

    parser.add_argument("--gpu", action="store_true", default=False)
    args = parser.parse_args()

    with open("config.json", "r") as f:
        jsonf = json.load(f)

    dt = Data(args, jsonf)
    print(f"Config file loaded: {dt.module.upper()} | {dt.dataset.upper()}")

    if(args.preprocess):
        dt.imread_dataset()
        dt.preprocess()
        dt.imwrite()

    elif(args.train):
        dt.imread_train()
        # model = Model(...)
        # train(model, dt.train)

    elif(args.test):
        dt.imread_test()
        # model = Model(...)
        # test(model, dt.test)


if __name__ == '__main__':
    main()
