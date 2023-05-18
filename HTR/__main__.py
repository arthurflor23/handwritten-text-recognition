import os
import tasks
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Task flags
    parser.add_argument('--check', default=False, action='store_true',
                        help="Perform data verification")
    parser.add_argument('--infer', default=False, action='store_true',
                        help="Perform inference process")
    parser.add_argument('--test', default=False, action='store_true',
                        help="Perform optical model test")
    parser.add_argument('--train', default=False, action='store_true',
                        help="Perform optical model training")

    # Optical Model
    parser.add_argument('--optical-model', default=None,
                        help="Define the optical model (bluche, puigcerver, flor)")

    # Dataset
    parser.add_argument('--source', default=None,
                        help="Define the source data (iam, rimes)")
    parser.add_argument('--level', default='line',
                        help="Define the recoginition level (line, paragraph)")
    parser.add_argument('--lazy-mode', default=False, action='store_true',
                        help="Enable lazy loading")
    parser.add_argument('--train-ratio', default=None,
                        help="Set the training partition ratio")
    parser.add_argument('--validation-ratio', default=None,
                        help="Set the validation partition ratio")
    parser.add_argument('--test-ratio', default=None,
                        help="Set the test partition ratio")

    # Data augmentation
    parser.add_argument('--disable-aug', default=True, action='store_false',
                        help="Disable data augmentation completely")
    parser.add_argument('--aug-rotation', default=[-1.5, 1.5], nargs='+',
                        help="Set rotation transformation (min_value, max_value)")

    # Inference
    parser.add_argument('--images', default=[], nargs='+',
                        help="Set image path list for handwriting recognition")
    parser.add_argument('--crop', default=[], nargs='+',
                        help="Set cropping values (x, y, width, height)")

    # Spell check
    parser.add_argument('--spell-check', default='openai',
                        help="Define the spell check (openai)")
    parser.add_argument('--api-key', default=None,
                        help="Set the spell check API_KEY")

    # Others
    parser.add_argument('--id', default=None,
                        help="Specify running id")

    args = parser.parse_args()

    # Setup basic paths
    args.base_path = os.path.join(os.path.dirname(__file__), '..')
    args.nltk_path = os.path.join(args.base_path, 'mlruns', 'nltk')
    args.output_path = os.path.join(args.base_path, 'mlruns')
    args.input_path = os.path.join(args.base_path, 'data')

    # Turn required parameters
    if args.check or args.train or args.test:
        assert args.source is not None, "source must be defined"

    if args.train or args.test or args.infer:
        assert args.network is not None, "network must be defined"

    # Forward to tasks
    if args.check:
        tasks.check(args)

    else:
        if args.train:
            tasks.train(args)

        if args.test:
            tasks.test(args)

        if args.infer and len(args.images):
            tasks.infer(args)
