import tasks
import argparse


if __name__ == '__main__':
    """
    Main entry point of the program. It parses the command-line arguments and
    triggers the appropriate task (train, test, infer, or check) based on those arguments.
    """

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--source', default=None,
                        help="Define the source data (iam, rimes)")

    parser.add_argument('--level', default='line',
                        help="Define the recoginition level (line, paragraph)")

    parser.add_argument('--training-ratio', default=None,
                        help="Set the training partition ratio")

    parser.add_argument('--validation-ratio', default=None,
                        help="Set the validation partition ratio")

    parser.add_argument('--test-ratio', default=None,
                        help="Set the test partition ratio")

    parser.add_argument('--lazy-mode', default=False, action='store_true',
                        help="Enable lazy loading")

    # Data augmentation
    parser.add_argument('--elastic-transform', default=[0.99, 43, 0.75], nargs='+', type=float,
                        help="Apply elastic transform (probability, kernel size, alpha)")

    parser.add_argument('--erosion', default=[0.99, 5, 1], nargs='+', type=float,
                        help="Apply dilation (probability, kernel size, iterations)")

    parser.add_argument('--dilation', default=[0.99, 2, 1], nargs='+', type=float,
                        help="Apply dilation (probability, kernel size, iterations)")

    parser.add_argument('--mixup', default=[0.99, 0.3, 1], nargs='+', type=float,
                        help="Apply mixup augmentation (probability, opacity, iterations)")

    parser.add_argument('--perspective-transform', default=[0.99, 0.4], nargs='+', type=float,
                        help="Apply perspective transformation (probability, alpha)")

    parser.add_argument('--salt-and-pepper', default=[0.99, 0.3], nargs='+', type=float,
                        help="Apply Gaussian noise (probability, alpha)")

    parser.add_argument('--gaussian-blur', default=[0.99, 11, 1], nargs='+', type=float,
                        help="Apply Gaussian blur (probability, kernel size, iterations)")

    parser.add_argument('--shearing', default=[0.99, 30], nargs='+', type=float,
                        help="Apply shearing transformation (probability, angle)")

    parser.add_argument('--scaling', default=[0.99, 0.05], nargs='+', type=float,
                        help="Apply scaling transformation (probability, scale alpha)")

    parser.add_argument('--rotation', default=[0.99, 0.5], nargs='+', type=float,
                        help="Apply rotation transformation (probability, angle)")

    parser.add_argument('--translation', default=[0.99, 0.025, 0.05], nargs='+', type=float,
                        help="Apply vertical and horizontal translation (probability, y-alpha, x-alpha)")

    parser.add_argument('--disable-augmentation', default=False, action='store_true',
                        help="Disable data augmentation completely")

    # Optical Network
    parser.add_argument('--network', default=None,
                        help="Define the optical network (bluche, flor, puigcerver)")

    # Spell checker
    parser.add_argument('--spell-checker', default=None,
                        help="Define the spell check (openai)")

    parser.add_argument('--api-key', default=None,
                        help="Set the spell checker API_KEY directly")

    parser.add_argument('--env-key', default=None,
                        help="Define the environment variable which holds the API key")

    # Training
    parser.add_argument('--train', default=False, action='store_true',
                        help="Perform optical model training")

    parser.add_argument('--epochs', default=1000, type=int,
                        help="Epochs for the training")

    parser.add_argument('--batch-size', default=16, type=int,
                        help="Batch size for the generator")

    parser.add_argument('--learning-rate', default=1e-3, type=float,
                        help="Learning rate for the optimizer")

    parser.add_argument('--plateau-factor', default=0.5, type=float,
                        help="Factor by which the learning rate will be reduced on a plateau")

    parser.add_argument('--plateau-cooldown', default=10, type=int,
                        help="Cooldown period after a learning rate plateau is triggered")

    parser.add_argument('--plateau-patience', default=20, type=int,
                        help="Number of epochs without improvement for the learning rate to be reduced")

    parser.add_argument('--patience', default=60, type=int,
                        help="Number of epochs with no improvement after which training will be stopped")

    # Test
    parser.add_argument('--test', default=False, action='store_true',
                        help="Perform optical model test")

    parser.add_argument('--beam-width', default=30, type=int,
                        help="The width of the beam for the CTC decoder")

    parser.add_argument('--top-paths', default=1, type=int,
                        help="Number of top paths to prediction")

    parser.add_argument('--share-top-paths', default=False, action='store_true',
                        help="Consider previous paths to the metrics")

    # Inference
    parser.add_argument('--infer', default=False, action='store_true',
                        help="Perform inference process")

    parser.add_argument('--images', default=[], nargs='+',
                        help="Set image path list for handwriting recognition")

    parser.add_argument('--bbox', default=[], nargs='+',
                        help="Set bounding box values (x, y, width, height)")

    # Check
    parser.add_argument('--check', default=False, action='store_true',
                        help="Perform data verification")

    # Others
    parser.add_argument('--experiment-name', default='Default',
                        help="Define MLflow experiment name")

    parser.add_argument('--run-index', default=None, type=int,
                        help="Specify run index")

    args = parser.parse_args()

    # Turn required parameters
    if args.check or args.train or args.test:
        assert args.source is not None, "source must be defined"

    if args.train or args.test or args.infer:
        assert args.network is not None, "network must be defined"

    if args.test or args.infer:
        assert args.run_index is not None, "run index must be defined"

    if args.infer:
        assert len(args.images) > 0, "images must be defined"

    # Tasks
    if args.train:
        tasks.train(args)

    elif args.test:
        tasks.test(args)

    elif args.infer:
        tasks.infer(args)

    elif args.check:
        tasks.check(args)
