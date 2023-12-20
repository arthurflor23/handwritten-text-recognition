import os
import sys
import argparse
import pipelines

from models.graphite import Graphite


if __name__ == '__main__':
    """
    It parses the command-line arguments and triggers the appropriate pipeline.
    """

    parser = argparse.ArgumentParser()

    # models
    parser.add_argument('--workflow', default='recognition', help='Define workflow')
    parser.add_argument('--synthesis', default='gan', help='Define synthesis model')
    parser.add_argument('--recognition', default='flor', help='Define recognition model')
    parser.add_argument('--spelling', default=None, help='Define spelling model')

    # mlflow
    parser.add_argument('--synthesis-index', default=None, type=int, help='Define a synthesis run index')
    parser.add_argument('--recognition-index', default=None, type=int, help='Define a recognition run index')
    parser.add_argument('--experiment-name', default='Default', help='Define an experiment name')

    # dataset
    parser.add_argument('--source', default=None, help='Define source data')
    parser.add_argument('--text-level', default='line', help='Define text structure level')
    parser.add_argument('--image-shape', default=[1024, 128, 1], nargs=3, type=int, help='Define image shape (WxHxC)')
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
    parser.add_argument('--disable-augmentation', default=False, action='store_true', help='Disable augmentation')

    # synthesis
    parser.add_argument('--synthesis-ratio', default=1.0, type=float, help='Probability to use synthetic data')

    # training
    parser.add_argument('--training', default=False, action='store_true', help='Perform training pipeline')
    parser.add_argument('--epochs', default=None, type=int, help='Maximum number of epochs')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size')
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='Optimizer learning rate')
    parser.add_argument('--plateau-factor', default=0.1, type=float, help='Learning rate reduction factor')
    parser.add_argument('--plateau-cooldown', default=0, type=int, help='Cooldown after rate plateau')
    parser.add_argument('--plateau-patience', default=20, type=int, help='Epochs before recognizing a plateau')
    parser.add_argument('--patience', default=30, type=int, help='Epochs without improvement to stop')

    # test
    parser.add_argument('--test', default=False, action='store_true', help='Perform test pipeline')
    parser.add_argument('--top-paths', default=1, type=int, help='Number of top paths to prediction')
    parser.add_argument('--beam-width', default=15, type=int, help='Beam width for CTC decoder')

    # inference
    parser.add_argument('--inference', default=False, action='store_true', help='Perform inference pipeline')
    parser.add_argument('--images', default=[], nargs='+', help='List of image paths')
    parser.add_argument('--bbox', default=[], nargs='+', help='Bounding box values for images (x, y, width, height)')
    parser.add_argument('--texts', default=[], nargs='+', help='List of arbitrary text inputs')

    # others
    parser.add_argument('--check', default=False, action='store_true', help='Perform check pipeline')
    parser.add_argument('--seed', default=1234, type=int, help='Seed value')

    args = parser.parse_args()

    # jupyter notebook compatibility
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    # mlflow compatibility
    Graphite().fix_mlflow_artifacts_path()

    # required parameters
    if args.check or args.training or args.test:
        assert args.source, '--source must be defined'

    if 'synthesis' in args.workflow:
        assert args.synthesis, '--synthesis must be defined'

        if args.inference:
            assert len(args.texts) > 0, '--texts must be defined'

    if 'recognition' in args.workflow:
        assert args.recognition, '--recognition must be defined'

        if args.inference:
            assert len(args.images) > 0, '--images must be defined'

    if args.training or args.test or args.inference:
        assert args.synthesis or args.recognition, '--synthesis or --recognition must be defined'

    if args.test or args.inference:
        assert args.run_index is not None, '--run-index must be defined'

    # pipelines
    if args.check:
        pipelines.check(args)

    elif args.training or args.test:
        pipelines.run(args, training=args.training)

    elif args.inference:
        pipelines.inference(args)
