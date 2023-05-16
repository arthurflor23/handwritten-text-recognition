import os
import json
import tasks
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', default=None, help="define source data for training and testing phases")
    parser.add_argument('--level', default=None, help="define recognition level (character, word, line, paragraph)")
    # parser.add_argument('--lazy-mode', default=False, action='store_true', help="define data loading on demand")
    # parser.add_argument('--source-metadata', default={}, type=json.loads, help="metadata object for source module")

    # parser.add_argument('--network', default=None, help="define network architecture")
    # parser.add_argument('--network-metadata', default={}, type=json.loads, help="metadata object for network module")

    # parser.add_argument('--train', default=False, action='store_true', help="perform source training phase")
    # parser.add_argument('--train-metadata', default={}, type=json.loads, help="metadata object for train module")

    # parser.add_argument('--test', default=False, action='store_true', help="perform source test phase")
    # parser.add_argument('--test-metadata', default={}, type=json.loads, help="metadata object for test module")

    # parser.add_argument('--infer', default=[], nargs='+', help="recognize handwritten text in list of arbitrary images")
    # parser.add_argument('--infer-metadata', default={}, type=json.loads, help="metadata object for infer module")

    # parser.add_argument('--check', default=None, help="validate data (source, images, labels, augmentation, encoding)")

    args = parser.parse_args()

    args.data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    args.output_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts')

    # if args.check is not None:
    #     assert args.source is not None, "source must be defined"

    #     if args.check == 'augmentation':
    #         tasks.check.augmentation(args)

    #     elif args.check == 'encoding':
    #         tasks.check.encoding(args)

    #     elif args.check == 'images':
    #         tasks.check.images(args)

    #     elif args.check == 'labels':
    #         tasks.check.labels(args)

    #     elif args.check == 'source':
    #         tasks.check.source(args)

    # else:
    #     assert args.network is not None, "network must be defined"

    #     if args.train:
    #         assert args.source is not None, "source must be defined"
    #         tasks.train(args)

    #     if args.test:
    #         assert args.source is not None, "source must be defined"
    #         tasks.test(args)

    #     if args.infer and len(args.infer):
    #         tasks.infer(args)
