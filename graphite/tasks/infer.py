import cv2
import numpy as np

from dataset import Dataset
from model import Model
from spelling import Spelling


def infer(args):
    """
    Performs inference phase.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object that contains all the arguments required.
    """

    def print_section(content):
        print(f"\n{'=' * 72}\n{content}\n{'=' * 72}\n")

    data = [[image, args.bbox, ['']] for image in args.images]

    dataset = Dataset(data=data,
                      level=args.level,
                      binarization=args.binarization,
                      eager_mode=args.eager_mode,
                      seed=args.seed)

    if args.verbose:
        print_section(dataset)

    model = Model(network=args.network, experiment_name=args.experiment_name)
    model.compile(run_index=args.run_index)

    if args.verbose:
        print_section(model)

    infer_data, infer_steps = dataset.get_generator(partition=dataset.test,
                                                    batch_size=args.batch_size,
                                                    shuffle=False)

    predictions, probabilities = model.predict(test_data=infer_data,
                                               test_steps=infer_steps,
                                               top_paths=args.top_paths,
                                               beam_width=args.beam_width,
                                               ctc_decode=True,
                                               token_decode=True,
                                               verbose=args.verbose)

    if args.verbose:
        print_section(model.test_logger)

    if args.spell_checker:
        spelling = Spelling(spell_checker=args.spell_checker,
                            api_key=args.api_key,
                            env_key=args.env_key)

        if args.verbose:
            print_section(spelling)

        predictions = spelling.enhance(predictions, verbose=args.verbose)

    for i in range(dataset.test['size']):
        print("Path   Probability   Predict")

        for j in range(args.top_paths):
            probability = np.mean(probabilities[j][i])
            predict = '\n'.join(predictions[j][i])

            print(f"{j+1:>4}   {probability:>11.2%}   {predict}")

        if args.check:
            print("\nPress Enter to continue or Esc to stop...\n")

            cv2.imshow("Image", dataset.test['data'][i][0])
            key = cv2.waitKey(0)

            if key == 27:
                cv2.destroyAllWindows()
                return
