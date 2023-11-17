import cv2

from datasets import Dataset
from models import Model
from spelling import Spelling


def test(args):
    """
    Performs testing phase.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object that contains all the arguments required.
    """

    def print_section(content):
        print(f"\n{'=' * 72}\n{content}\n{'=' * 72}\n")

    dataset = Dataset(source=args.source,
                      level=args.level,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      binarization=args.binarization,
                      eager_mode=args.eager_mode,
                      seed=args.seed)

    if args.verbose > 0:
        print_section(dataset)

    model = Model(network=args.network, experiment_name=args.experiment_name, seed=args.seed)
    model.compile(tokenizer=dataset.tokenizer, learning_rate=args.learning_rate, run_index=args.run_index)

    if args.verbose > 1:
        print_section(model)

    test_data, test_steps = dataset.get_generator(partition=dataset.test,
                                                  batch_size=args.batch_size,
                                                  shuffle=False)

    predictions, _ = model.predict(test_data=test_data,
                                   test_steps=test_steps,
                                   top_paths=args.top_paths,
                                   beam_width=args.beam_width,
                                   ctc_decode=True,
                                   token_decode=True,
                                   verbose=args.verbose)

    if args.verbose > 0:
        print_section(model.test_logger)

    baseline_metrics = model.evaluate(partition=dataset.test,
                                      baseline_predictions=predictions,
                                      share_top_paths=args.share_top_paths)

    if not args.check:
        model.save_context(dataset=dataset, baseline_metrics=baseline_metrics)

    if args.spell_checker:
        spelling = Spelling(spell_checker=args.spell_checker,
                            api_key=args.api_key,
                            env_key=args.env_key)

        if args.verbose > 1:
            print_section(spelling)

        spelling_predictions = spelling.enhance(predictions, verbose=args.verbose)

        spelling_metrics = model.evaluate(partition=dataset.test,
                                          spelling_predictions=spelling_predictions,
                                          share_top_paths=args.share_top_paths)

        if not args.check:
            model.save_context(spelling=spelling, spelling_metrics=spelling_metrics)

    if args.verbose > 0:
        print_section(model.evaluation_logger)

    if args.check:
        print("\nChecking samples...\n")

        test_data, test_steps = dataset.get_generator(dataset.test)
        pred_index = 0

        for _ in range(test_steps):
            batch_data, batch_labels = next(test_data)

            for i in range(len(batch_data)):
                image = batch_data[i]
                label = dataset.tokenizer.decode(batch_labels[i])

                print("\nTest Label")
                print('\n'.join(label))

                print("\nPrediction")
                print('\n'.join(predictions[0, pred_index, :]))

                if args.spell_checker:
                    print("\nSpelling Prediction")
                    print('\n'.join(spelling_predictions[0, pred_index, :]))

                pred_index += 1

                print("\n\nPress Enter to continue or Esc to stop...\n")

                cv2.imshow("Image", image)
                key = cv2.waitKey(0)

                if key == 27:
                    cv2.destroyAllWindows()
                    return
