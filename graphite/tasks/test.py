import cv2

from dataset import Dataset
from model import Model
from spelling import Spelling


def test(args):
    """
    Performs testing phase.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object that contains all the arguments required.
    """

    dataset = Dataset(source=args.source,
                      level=args.level,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      lazy_mode=args.lazy_mode,
                      seed=42)

    print(dataset)

    model = Model(network=args.network, experiment_name=args.experiment_name, seed=42)

    model.compile(tokenizer=dataset.tokenizer, learning_rate=args.learning_rate)
    model.load_context(run_index=args.run_index)

    print(model)

    test_data, test_steps = dataset.get_generator(dataset.test, batch_size=args.batch_size, shuffle=False)

    predictions, _ = model.predict(test_data=test_data,
                                   test_steps=test_steps,
                                   top_paths=args.top_paths,
                                   beam_width=args.beam_width,
                                   ctc_decode=True,
                                   token_decode=True,
                                   verbose=1)

    baseline_metrics, _ = model.evaluate(dataset.test,
                                         baseline_predictions=predictions,
                                         share_top_paths=args.share_top_paths)

    if not args.check:
        model.save_context(dataset=dataset, baseline_metrics=baseline_metrics)

    if args.spell_checker:
        spelling = Spelling(spell_checker=args.spell_checker,
                            api_key=args.api_key,
                            env_key=args.env_key)

        spelling_predictions = spelling.enhance(predictions)

        spelling_metrics, _ = model.evaluate(dataset.test,
                                             spelling_predictions=spelling_predictions,
                                             share_top_paths=args.share_top_paths)

        if not args.check:
            model.save_context(spelling=spelling, spelling_metrics=spelling_metrics)

    print(model.test_logger)

    if args.check:
        print("\nChecking samples...\n")

        test_data, test_steps = dataset.get_generator(dataset.test)
        pred_index = 0

        for _ in range(test_steps):
            batch_data, batch_labels = next(test_data)

            for i in range(len(batch_data)):
                image = batch_data[i]
                label = dataset.tokenizer.decode(batch_labels[i])

                cv2.imshow("Test Image", image)

                print("\nTest Label")
                print('\n'.join(label))

                print("\nPrediction")
                print('\n'.join(predictions[0, pred_index, :]))

                if args.spell_checker:
                    print("\nSpelling Prediction")
                    print('\n'.join(spelling_predictions[0, pred_index, :]))

                pred_index += 1

                print("\n\nPress Enter to continue or Esc to stop...\n")
                key = cv2.waitKey(0)

                if key == 27:
                    cv2.destroyAllWindows()
                    return
