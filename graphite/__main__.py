import task
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Task flags
    parser.add_argument('--infer', default=False, action='store_true',
                        help="Perform inference process")

    parser.add_argument('--test', default=False, action='store_true',
                        help="Perform optical model test")
    parser.add_argument('--train', default=False, action='store_true',
                        help="Perform optical model training")

    parser.add_argument('--check', default=False, action='store_true',
                        help="Perform data verification")
    parser.add_argument('--check-samples', default=False, action='store_true',
                        help="View sample data")

    # Optical Model
    parser.add_argument('--optical-model', default=None,
                        help="Define the optical model (bluche, flor, puigcerver)")

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
    parser.add_argument('--erosion', default=[0.99, 5, 2], nargs='+', type=float,
                        help="Apply dilation (probability, kernel_size, iterations)")
    parser.add_argument('--dilation', default=[0.99, 3, 1], nargs='+', type=float,
                        help="Apply dilation (probability, kernel_size, iterations)")
    parser.add_argument('--elastic-transform', default=[0.99, 33], nargs='+', type=float,
                        help="Apply elastic transform (probability, kernel size)")
    parser.add_argument('--perspective-transform', default=[0.99, 0.5], nargs='+', type=float,
                        help="Apply perspective transformation (probability, factor)")
    parser.add_argument('--mixup', default=[0.99, 0.2, 1], nargs='+', type=float,
                        help="Apply mixup augmentation (probability, opacity, pickups)")
    parser.add_argument('--shearing', default=[0.99, 1.5], nargs='+', type=float,
                        help="Apply shearing transformation (probability, factor)")
    parser.add_argument('--scaling', default=[0.99, 0.5, 1.5], nargs='+', type=float,
                        help="Apply scaling transformation (probability, min_factor, max_factor)")
    parser.add_argument('--rotation', default=[0.99, 5], nargs='+', type=float,
                        help="Apply rotation transformation (probability, angle)")
    parser.add_argument('--translation', default=[0.99, 0.5, 0.5], nargs='+', type=float,
                        help="Apply vertical and horizontal translation (probability, y_factor, x_factor)")
    parser.add_argument('--salt-and-pepper', default=[0.99, 0.1], nargs='+', type=float,
                        help="Apply Gaussian noise (probability, percentage)")
    parser.add_argument('--gaussian-blur', default=[0.99, 3, 5], nargs='+', type=float,
                        help="Apply Gaussian blur (probability, kernel size, iterations)")
    parser.add_argument('--disable-augmentation', default=True, action='store_false',
                        help="Disable data augmentation completely")

    # Inference
    parser.add_argument('--images', default=[], nargs='+',
                        help="Set image path list for handwriting recognition")
    parser.add_argument('--bbox', default=[], nargs='+',
                        help="Set bounding box values (x, y, width, height)")

    # Spell check
    parser.add_argument('--spell-check', default='openai',
                        help="Define the spell check (openai)")
    parser.add_argument('--api-key', default=None,
                        help="Set the spell check API_KEY")

    # Others
    parser.add_argument('--id', default=None,
                        help="Specify running id")

    args = parser.parse_args()

    # Turn required parameters
    if args.check or args.train or args.test:
        assert args.source is not None, "source must be defined"

    if args.train or args.test or args.infer:
        assert args.network is not None, "network must be defined"

    # Tasks
    if args.check:
        task.check(args)

    else:
        if args.train:
            task.train(args)

        if args.test:
            task.test(args)

        if args.infer and len(args.images):
            task.infer(args)
