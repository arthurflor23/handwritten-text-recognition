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
    parser.add_argument('--synthesis', default=None, help='Synthesis model')
    parser.add_argument('--recognition', default=None, help='Recognition model')
    parser.add_argument('--spelling', default=None, help='Spelling model')

    # mlflow
    parser.add_argument('--synthesis-run-id', default=None, help='Synthesis run id or index')
    parser.add_argument('--recognition-run-id', default=None, help='Recognition run id or index')
    parser.add_argument('--experiment-name', default='Default', help='Experiment name')
    parser.add_argument('--all-runs', default=False, action='store_true', help='Enable all runs for selection')

    # dataset
    parser.add_argument('--source', default=None, help='Source data')
    parser.add_argument('--source-input-path', default='datasets', help='Source input path')
    parser.add_argument('--text-level', default='line', help='Text structure level')
    parser.add_argument('--image-shape', default=[1024, 64, 1], nargs=3, type=int, help='Image dimensions (w, h, c)')
    parser.add_argument('--char-width', default=None, type=int, help='Character width for normalization')
    parser.add_argument('--training-ratio', default=None, help='Training partition ratio')
    parser.add_argument('--validation-ratio', default=None, help='Validation partition ratio')
    parser.add_argument('--test-ratio', default=None, help='Test partition ratio')
    parser.add_argument('--lazy-mode', default=False, action='store_true', help='Enable lazy loading')

    # augmentor
    parser.add_argument('--binarize', default=None, nargs='+', help='Binarization settings')
    parser.add_argument('--erode', default=[0.66, 3], nargs=2, type=float, help='Erosion settings')
    parser.add_argument('--dilate', default=[0.33, 2], nargs=2, type=float, help='Dilation settings')
    parser.add_argument('--elastic', default=[0.66, 7], nargs=2, type=float, help='Elastic deformation settings')
    parser.add_argument('--perspective', default=[0.33, 0.5], nargs=2, type=float, help='Perspective settings')
    parser.add_argument('--mixup', default=None, nargs=2, type=float, help='Mixup settings')
    parser.add_argument('--shear', default=[0.66, 15], nargs=2, type=float, help='Shearing settings')
    parser.add_argument('--scale', default=[0.66, 0.05], nargs=2, type=float, help='Scaling settings')
    parser.add_argument('--rotate', default=[0.66, 0.1], nargs=2, type=float, help='Rotation settings')
    parser.add_argument('--shift-y', default=[0.66, 0.05], nargs=2, type=float, help='Vertical shift settings')
    parser.add_argument('--shift-x', default=[0.33, 0.01], nargs=2, type=float, help='Horizontal shift settings')
    parser.add_argument('--salt-and-pepper', default=None, nargs=2, type=float, help='Salt and pepper noise settings')
    parser.add_argument('--gaussian-noise', default=None, nargs=2, type=float, help='Gaussian noise settings')
    parser.add_argument('--gaussian-blur', default=None, nargs=2, type=float, help='Gaussian blur settings')
    parser.add_argument('--disable-augmentation', default=False, action='store_true', help='Disable all augmentations')

    # synthesis
    parser.add_argument('--discriminator-steps', default=1, type=int, help='Discriminator step repetitions in training')
    parser.add_argument('--generator-steps', default=1, type=int, help='Generator step skips in training')

    # recognition
    parser.add_argument('--synthetic-data-ratio', default=1.0, type=float, help='Synthetic data ratio for training')

    # training
    parser.add_argument('--training', default=False, action='store_true', help='Perform training pipeline')
    parser.add_argument('--epochs', default=None, type=int, help='Maximum number of epochs')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size')
    parser.add_argument('--learning-rate', default=None, type=float, help='Learning rate')
    parser.add_argument('--plateau-factor', default=0.1, type=float, help='Learning rate reduction factor')
    parser.add_argument('--plateau-cooldown', default=0, type=int, help='Cooldown after plateau')
    parser.add_argument('--plateau-patience', default=20, type=int, help='Plateau patience epochs')
    parser.add_argument('--patience', default=40, type=int, help='Stop after no improvement')

    # test
    parser.add_argument('--test', default=False, action='store_true', help='Perform test pipeline')
    parser.add_argument('--top-paths', default=1, type=int, help='Top paths for prediction')
    parser.add_argument('--beam-width', default=32, type=int, help='CTC decoder beam width')

    # inference
    parser.add_argument('--inference', default=False, action='store_true', help='Perform inference pipeline')
    parser.add_argument('--inference-output-path', default='.', help='Inference output path')
    parser.add_argument('--image', default=None, help='Image path for recognition')
    parser.add_argument('--bbox', default=None, nargs=4, help='Bounding box (x, y, w, h)')
    parser.add_argument('--text', default=None, help='Text for synthesis')

    # others
    parser.add_argument('--check', default=False, action='store_true', help='Perform check pipeline')
    parser.add_argument('--gpu', default=0, type=str, help='GPU index value')
    parser.add_argument('--seed', default=0, type=int, help='Seed value')

    args = parser.parse_args()
    args.gpu = int(args.gpu) if str(args.gpu).isdigit() else None

    # required
    if args.check or args.training or args.test:
        assert args.source, '--source must be defined'

    if args.training or args.test or args.inference:
        assert args.synthesis or args.recognition, '--synthesis or --recognition must be defined'

    elif args.spelling:
        assert args.recognition_run_id, '--recognition-run-id must be defined'

    if args.synthesis and args.inference:
        assert args.text, '--text must be defined'

    if args.recognition and args.inference:
        assert args.image, '--image must be defined'

    # pipelines
    if args.check:
        pipelines.check(args)

    elif args.inference:
        pipelines.inference(args)

    elif args.training or args.test or args.spelling:
        pipelines.run(args)

    else:
        # mlflow path compatibility
        #   https://github.com/mlflow/mlflow/issues/3144
        from sarah.models.compose import Compose
        Compose().fix_mlflow_artifacts_path()
