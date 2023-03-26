import os
import json
import tasks
import argparse


if __name__ == '__main__':
    # setup parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', default=None, help="source data to load (training/testing)")
    parser.add_argument('--source-metadata', default={}, type=json.loads, help="metadata object for source module")

    parser.add_argument('--network', default=None, help="network architecture for recognition")
    parser.add_argument('--network-metadata', default={}, type=json.loads, help="metadata object for network module")

    parser.add_argument('--train', default=False, action='store_true', help="flag to perform model training")
    parser.add_argument('--train-metadata', default={}, type=json.loads, help="metadata object for train module")

    parser.add_argument('--test', default=False, action='store_true', help="flag to perform model test")
    parser.add_argument('--test-metadata', default={}, type=json.loads, help="metadata object for test module")

    parser.add_argument('--infer', default=[], nargs='+', help="image path list for recognition")
    parser.add_argument('--infer-metadata', default={}, type=json.loads, help="metadata object for infer module")

    parser.add_argument('--check-source', default=False, action='store_true', help="validate source data")
    parser.add_argument('--check-augmentation', default=False, action='store_true', help="visualize data augmentation")

    args = parser.parse_args()

    # setup default paths
    args.data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    args.output_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts')

    # setup tasks
    if args.check_source or args.check_augmentation:
        if args.check_source:
            tasks.check_source(args)

        if args.check_augmentation:
            tasks.check_augmentation(args)

    else:
        if args.train:
            tasks.train(args)

        if args.test:
            tasks.test(args)

        if args.infer and len(args.infer):
            tasks.infer(args)
