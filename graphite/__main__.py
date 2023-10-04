import os
import sys
import tasks
import argparse


if __name__ == '__main__':
    """
    It parses the command-line arguments and triggers the appropriate task.
    """

    parser = argparse.ArgumentParser()

    # Models
    parser.add_argument('--generative', default=None, help='Define generative model')
    parser.add_argument('--optical', default=None, help='Define optical model')
    parser.add_argument('--run-index', default=None, type=int, help='Define run index')

    # Dataset
    parser.add_argument('--source', default=None, help='Define source data')
    parser.add_argument('--level', default='line', help='Define text level')
    parser.add_argument('--shape', default=[128, 1024, 1], nargs='+', type=int, help='Define image shape')
    parser.add_argument('--training-ratio', default=None, help='Define training partition ratio')
    parser.add_argument('--validation-ratio', default=None, help='Define validation partition ratio')
    parser.add_argument('--test-ratio', default=None, help='Define test partition ratio')
    parser.add_argument('--binarization', default=False, action='store_true', help='Enable binarization')
    parser.add_argument('--lazy-mode', default=False, action='store_true', help='Enable lazy loading mode')

    # Augmentor
    parser.add_argument('--erode', default=[0.66, 3, 1], nargs='+', type=float, help='Erosion parameters')
    parser.add_argument('--dilate', default=[0.33, 2, 1], nargs='+', type=float, help='Dilation parameters')
    parser.add_argument('--elastic', default=[0.66, 29, 1.0], nargs='+', type=float, help='Elastic parameters')
    parser.add_argument('--perspective', default=[0.66, 0.4], nargs='+', type=float, help='Perspective parameters')
    parser.add_argument('--mixup', default=None, nargs='+', type=float, help='Mixup parameters')
    parser.add_argument('--shear', default=[0.33, 15], nargs='+', type=float, help='Shearing parameters')
    parser.add_argument('--scale', default=[0.33, 0.1], nargs='+', type=float, help='Scaling parameters')
    parser.add_argument('--rotate', default=[0.33, 1.0], nargs='+', type=float, help='Rotation parameters')
    parser.add_argument('--shift-y', default=None, nargs='+', type=float, help='Vertical translation parameters')
    parser.add_argument('--shift-x', default=None, nargs='+', type=float, help='Horizontal translation parameters')
    parser.add_argument('--salt-and-pepper', default=None, nargs='+', type=float, help='Salt & pepper parameters')
    parser.add_argument('--gaussian-noise', default=None, nargs='+', type=float, help='Gaussian noise parameters')
    parser.add_argument('--gaussian-blur', default=None, nargs='+', type=float, help='Gaussian blur parameters')
    parser.add_argument('--disable-augmentation', default=False, action='store_true', help='Disable all augmentations')

    # Training
    parser.add_argument('--training', default=False, action='store_true', help='Perform generative or optical training')
    parser.add_argument('--epochs', default=1e6, type=int, help='Maximum number of epochs')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size')
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='Optimizer learning rate')
    parser.add_argument('--plateau-factor', default=0.1, type=float, help='Learning rate reduction factor')
    parser.add_argument('--plateau-cooldown', default=0, type=int, help='Cooldown after rate plateau')
    parser.add_argument('--plateau-patience', default=20, type=int, help='Epochs before recognizing a plateau')
    parser.add_argument('--patience', default=30, type=int, help='Epochs without improvement to stop')

    # Test
    parser.add_argument('--test', default=False, action='store_true', help='Perform generative or optical test')
    parser.add_argument('--top-paths', default=1, type=int, help='Number of top paths to prediction')
    parser.add_argument('--beam-width', default=30, type=int, help='Beam width for CTC decoder')
    parser.add_argument('--share-top-paths', default=False, action='store_true', help='Use previous paths in metrics')
    parser.add_argument('--spell-checker', default='openai', help='Spell checker type')

    # Inference
    parser.add_argument('--infer', default=False, action='store_true', help='Perform generative or optical inference')
    parser.add_argument('--images', default=[], nargs='+', help='List of image paths')
    parser.add_argument('--bbox', default=[], nargs='+', help='Bounding box values for images (x, y, width, height)')
    parser.add_argument('--texts', default=[], nargs='+', help='List of arbitrary text inputs')

    # Others
    parser.add_argument('--check', default=False, action='store_true', help='Verify data')
    parser.add_argument('--seed', default=None, type=int, help='Seed value')
    parser.add_argument('--verbose', default=1, type=int, help='Verbosity level')

    args = parser.parse_args()

    # Jupyter notebook compatibility
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    # Required parameters
    if args.check or args.training or args.test:
        assert args.source, '--source must be defined'

    if args.training or args.test or args.infer:
        assert args.generative or args.optical, '--generative or --optical must be defined'

    if args.test or args.infer:
        assert args.run_index is not None, '--run-index must be defined'

    if args.infer:
        assert len(args.images) > 0, 'images must be defined'

    # Tasks
    if args.check:
        tasks.check(args)

    # elif args.training:
    #     tasks.train(args)

    # elif args.test:
    #     tasks.test(args)

    # elif args.infer:
    #     tasks.infer(args)
