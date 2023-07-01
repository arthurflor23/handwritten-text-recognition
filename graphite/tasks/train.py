from dataset import Augmentor, Dataset
from model import Model
from spelling import Spelling


def train(args):
    """
    Performs training and testing phases.

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
                      seed=args.seed)

    if args.verbose:
        print(dataset)

    augmentor = None

    if not args.disable_augmentation:
        augmentor = Augmentor(erosion=args.erosion,
                              dilation=args.dilation,
                              elastic_transform=args.elastic_transform,
                              perspective_transform=args.perspective_transform,
                              mixup=args.mixup,
                              gaussian_noise=args.gaussian_noise,
                              gaussian_blur=args.gaussian_blur,
                              shearing=args.shearing,
                              scaling=args.scaling,
                              rotation=args.rotation,
                              translation=args.translation,
                              seed=args.seed)

        if args.verbose:
            print(augmentor)

    model = Model(network=args.network, experiment_name=args.experiment_name, seed=args.seed)
    model.compile(tokenizer=dataset.tokenizer, learning_rate=args.learning_rate, run_index=args.run_index)

    if args.verbose:
        print(model)

    train_data, train_steps = dataset.get_generator(partition=dataset.training,
                                                    batch_size=args.batch_size,
                                                    augmentor=augmentor)

    valid_data, valid_steps = dataset.get_generator(partition=dataset.validation,
                                                    batch_size=args.batch_size,
                                                    augmentor=None)

    model.fit(epochs=args.epochs,
              training_data=train_data,
              training_steps=train_steps,
              validation_data=valid_data,
              validation_steps=valid_steps,
              plateau_factor=args.plateau_factor,
              plateau_cooldown=args.plateau_cooldown,
              plateau_patience=args.plateau_patience,
              patience=args.patience,
              verbose=args.verbose)

    if args.verbose:
        print(model.training_logger)

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

    baseline_metrics = model.evaluate(partition=dataset.test,
                                      baseline_predictions=predictions,
                                      share_top_paths=args.share_top_paths)

    model.save_context(dataset=dataset, augmentor=augmentor, baseline_metrics=baseline_metrics)

    if args.spell_checker:
        spelling = Spelling(spell_checker=args.spell_checker,
                            api_key=args.api_key,
                            env_key=args.env_key)

        spelling_predictions = spelling.enhance(predictions, verbose=args.verbose)

        spelling_metrics = model.evaluate(partition=dataset.test,
                                          spelling_predictions=spelling_predictions,
                                          share_top_paths=args.share_top_paths)

        model.save_context(spelling=spelling, spelling_metrics=spelling_metrics)

    if args.verbose:
        print(model.test_logger)
