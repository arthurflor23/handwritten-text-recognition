import os
import sys
import argparse

sys.path.append(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import pipelines  # noqa: E402


if __name__ == '__main__':
    """
    It parses the command-line arguments and triggers the appropriate pipeline.
    """

    parser = argparse.ArgumentParser()

    # models
    parser.add_argument('--synthesis', default=None, help='Define synthesis model')
    parser.add_argument('--recognition', default=None, help='Define recognition model')
    parser.add_argument('--spelling', default=None, help='Define spelling model')

    # mlflow
    parser.add_argument('--synthesis-run-index', default=None, type=int, help='Define a synthesis run index')
    parser.add_argument('--recognition-run-index', default=None, type=int, help='Define a recognition run index')
    parser.add_argument('--status-finished', default=False, action='store_true', help='Restrict run index status')
    parser.add_argument('--experiment-name', default='Default', help='Define an experiment name')

    # dataset
    parser.add_argument('--source', default=None, help='Define source data')
    parser.add_argument('--source-input-path', default='datasets', help='Source input path')
    parser.add_argument('--text-level', default='line', help='Define text structure level')
    parser.add_argument('--image-shape', default=[1024, 64, 1], nargs=3, type=int, help='Define image shape (w, h, c)')
    parser.add_argument('--char-width', default=None, type=int, help='Define character width for normalization')
    parser.add_argument('--training-ratio', default=None, help='Define training partition ratio')
    parser.add_argument('--validation-ratio', default=None, help='Define validation partition ratio')
    parser.add_argument('--test-ratio', default=None, help='Define test partition ratio')
    parser.add_argument('--lazy-mode', default=False, action='store_true', help='Enable lazy loading mode')

    # augmentor
    parser.add_argument('--binarize', default=None, nargs='+', help='Binarization parameters')
    parser.add_argument('--erode', default=None, nargs=2, type=float, help='Erosion parameters')
    parser.add_argument('--dilate', default=None, nargs=2, type=float, help='Dilation parameters')
    parser.add_argument('--elastic', default=None, nargs=2, type=float, help='Elastic parameters')
    parser.add_argument('--perspective', default=None, nargs=2, type=float, help='Perspective parameters')
    parser.add_argument('--mixup', default=None, nargs=2, type=float, help='Mixup parameters')
    parser.add_argument('--shear', default=None, nargs=2, type=float, help='Shearing parameters')
    parser.add_argument('--scale', default=None, nargs=2, type=float, help='Scaling parameters')
    parser.add_argument('--rotate', default=None, nargs=2, type=float, help='Rotation parameters')
    parser.add_argument('--shift-y', default=None, nargs=2, type=float, help='Vertical translation parameters')
    parser.add_argument('--shift-x', default=None, nargs=2, type=float, help='Horizontal translation parameters')
    parser.add_argument('--salt-and-pepper', default=None, nargs=2, type=float, help='Salt & pepper parameters')
    parser.add_argument('--gaussian-noise', default=None, nargs=2, type=float, help='Gaussian noise parameters')
    parser.add_argument('--gaussian-blur', default=None, nargs=2, type=float, help='Gaussian blur parameters')
    parser.add_argument('--disable-augmentation', default=False, action='store_true', help='Disable augmentation')

    # synthesis
    parser.add_argument('--discriminator-steps', default=1, type=int, help='Define repetition of steps for training')
    parser.add_argument('--generator-steps', default=1, type=int, help='Define skipping steps for training')
    parser.add_argument('--synthesis-ratio', default=1.0, type=float, help='Define synthetic data ratio for training')

    # training
    parser.add_argument('--training', default=False, action='store_true', help='Perform training pipeline')
    parser.add_argument('--epochs', default=None, type=int, help='Maximum number of epochs')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size')
    parser.add_argument('--learning-rate', default=None, type=float, help='Optimizer learning rate')
    parser.add_argument('--plateau-factor', default=0.1, type=float, help='Learning rate reduction factor')
    parser.add_argument('--plateau-cooldown', default=0, type=int, help='Cooldown after rate plateau')
    parser.add_argument('--plateau-patience', default=20, type=int, help='Epochs before recognizing a plateau')
    parser.add_argument('--patience', default=40, type=int, help='Epochs without improvement to stop')

    # test
    parser.add_argument('--test', default=False, action='store_true', help='Perform test pipeline')
    parser.add_argument('--top-paths', default=1, type=int, help='Number of top paths for prediction')
    parser.add_argument('--beam-width', default=30, type=int, help='Beam width for CTC decoder')

    # inference
    parser.add_argument('--inference', default=False, action='store_true', help='Perform inference pipeline')
    parser.add_argument('--inference-output-path', default='outputs', help='Inference output path')
    parser.add_argument('--image', default=None, help='Define the image path for handwriting recognition')
    parser.add_argument('--bbox', default=None, nargs=4, help='Bounding box values for image (x, y, w, h)')
    parser.add_argument('--text', default=None, help='Define the text input for handwriting synthesis')

    # others
    parser.add_argument('--check', default=False, action='store_true', help='Perform check pipeline')
    parser.add_argument('--seed', default=42, type=int, help='Seed value')

    args = parser.parse_args()

    # required
    if args.check or args.training or args.test:
        assert args.source, '--source must be defined'

    if args.training or args.test or args.inference:
        assert args.synthesis or args.recognition, '--synthesis or --recognition must be defined'

    if args.synthesis and args.inference:
        assert args.text, '--text must be defined'

    if args.recognition and args.inference:
        assert args.image, '--image must be defined'

    # pipelines
    if args.check:
        pipelines.check(args)

    elif args.training or args.test:
        pipelines.run(args, training=args.training)

    elif args.inference:
        pipelines.inference(args)

    else:
        # mlflow path compatibility
        #   https://github.com/mlflow/mlflow/issues/3144
        from models.graphite import Graphite
        Graphite().fix_mlflow_artifacts_path()
