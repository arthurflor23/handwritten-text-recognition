import os
import sys
import argparse
import pipelines


if __name__ == '__main__':
    """
    It parses the command-line arguments and triggers the appropriate pipeline.
    """

    parser = argparse.ArgumentParser()

    # models
    parser.add_argument('--mode', default='recognition', help='Define application mode')
    parser.add_argument('--synthesis', default=None, help='Define synthesis model')
    parser.add_argument('--recognition', default=None, help='Define recognition model')
    parser.add_argument('--spelling', default=None, help='Define spelling model')
    parser.add_argument('--run-index', default=None, type=int, help='Define run index')

    # dataset
    parser.add_argument('--source', default=None, help='Define source data')
    parser.add_argument('--text-level', default='line', help='Define text structure level')
    parser.add_argument('--image-shape', default=[128, 1024, 1], nargs=3, type=int, help='Define image shape (HxWxC)')
    parser.add_argument('--training-ratio', default=None, help='Define training partition ratio')
    parser.add_argument('--validation-ratio', default=None, help='Define validation partition ratio')
    parser.add_argument('--test-ratio', default=None, help='Define test partition ratio')
    parser.add_argument('--lazy-mode', default=False, action='store_true', help='Enable lazy loading mode')

    # augmentor
    parser.add_argument('--binarize', default=None, nargs='+', help='Binarization parameters')
    parser.add_argument('--erode', default=[0.66, 3, 1], nargs=3, type=float, help='Erosion parameters')
    parser.add_argument('--dilate', default=[0.33, 2, 1], nargs=3, type=float, help='Dilation parameters')
    parser.add_argument('--elastic', default=[0.66, 29, 1.0], nargs=3, type=float, help='Elastic parameters')
    parser.add_argument('--perspective', default=[0.66, 0.4], nargs=2, type=float, help='Perspective parameters')
    parser.add_argument('--mixup', default=None, nargs=3, type=float, help='Mixup parameters')
    parser.add_argument('--shear', default=[0.33, 15], nargs=2, type=float, help='Shearing parameters')
    parser.add_argument('--scale', default=[0.33, 0.1], nargs=2, type=float, help='Scaling parameters')
    parser.add_argument('--rotate', default=[0.33, 1.0], nargs=2, type=float, help='Rotation parameters')
    parser.add_argument('--shift-y', default=None, nargs=2, type=float, help='Vertical translation parameters')
    parser.add_argument('--shift-x', default=None, nargs=2, type=float, help='Horizontal translation parameters')
    parser.add_argument('--salt-and-pepper', default=None, nargs=2, type=float, help='Salt & pepper parameters')
    parser.add_argument('--gaussian-noise', default=None, nargs=2, type=float, help='Gaussian noise parameters')
    parser.add_argument('--gaussian-blur', default=None, nargs=3, type=float, help='Gaussian blur parameters')
    parser.add_argument('--disable-augmentation', default=False, action='store_true', help='Disable all augmentations')

    # training
    parser.add_argument('--training', default=False, action='store_true', help='Perform training pipeline')
    parser.add_argument('--epochs', default=1000000, type=int, help='Maximum number of epochs')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size')
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='Optimizer learning rate')
    parser.add_argument('--plateau-factor', default=0.1, type=float, help='Learning rate reduction factor')
    parser.add_argument('--plateau-cooldown', default=0, type=int, help='Cooldown after rate plateau')
    parser.add_argument('--plateau-patience', default=20, type=int, help='Epochs before recognizing a plateau')
    parser.add_argument('--patience', default=30, type=int, help='Epochs without improvement to stop')

    # test
    parser.add_argument('--test', default=False, action='store_true', help='Perform test pipeline')
    parser.add_argument('--top-paths', default=1, type=int, help='Number of top paths to prediction')
    parser.add_argument('--beam-width', default=30, type=int, help='Beam width for CTC decoder')
    parser.add_argument('--share-top-paths', default=False, action='store_true', help='Use previous paths in metrics')

    # inference
    parser.add_argument('--inference', default=False, action='store_true', help='Perform inference pipeline')
    parser.add_argument('--images', default=[], nargs='+', help='List of image paths')
    parser.add_argument('--bbox', default=[], nargs='+', help='Bounding box values for images (x, y, width, height)')
    parser.add_argument('--texts', default=[], nargs='+', help='List of arbitrary text inputs')

    # others
    parser.add_argument('--check', default=False, action='store_true', help='Perform check pipeline')
    parser.add_argument('--seed', default=1234, type=int, help='Seed value')
    parser.add_argument('--verbose', default=1, type=int, help='Verbosity level')

    args = parser.parse_args()

    # jupyter notebook compatibility
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    # required parameters
    if args.check or args.training or args.test:
        assert args.source, "--source must be defined"

    if args.training or args.test or args.inference:
        assert args.synthesis or args.recognition, "--synthesis or --recognition must be defined"

    if args.test or args.inference:
        assert args.run_index is not None, "--run-index must be defined"

    if args.inference and args.mode == 'synthesis':
        assert len(args.texts) > 0, "--texts must be defined"

    if args.inference and args.mode == 'recognition':
        assert len(args.images) > 0, "--images must be defined"

    # pipelines
    if args.training:
        pipelines.training(args)

    elif args.test:
        pipelines.test(args)

    elif args.inference:
        pipelines.inference(args)

    else:
        pipelines.check(args)
