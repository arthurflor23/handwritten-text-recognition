from dataset import Augmentor, Dataset
from model import Model
from spelling import Spelling


def train(args):
    """
    Trains a Model using a given Dataset and Augmentor.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object that contains all the arguments required for training.
    """

    dataset = Dataset(source=args.source,
                      level=args.level,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      lazy_mode=args.lazy_mode,
                      seed=42)

    print(dataset)

    augmentor = Augmentor(elastic_transform=args.elastic_transform,
                          erosion=args.erosion,
                          dilation=args.dilation,
                          mixup=args.mixup,
                          perspective_transform=args.perspective_transform,
                          salt_and_pepper=args.salt_and_pepper,
                          gaussian_blur=args.gaussian_blur,
                          shearing=args.shearing,
                          scaling=args.scaling,
                          rotation=args.rotation,
                          translation=args.translation,
                          disable_augmentation=args.disable_augmentation,
                          seed=42)

    print(augmentor)

    model = Model(network=args.network, experiment_name=args.experiment_name, seed=42)
    model.compile(tokenizer=dataset.tokenizer, learning_rate=args.learning_rate)

    print(model)

    if args.run_index is not None:
        model.load_context(run_index=args.run_index)

    train_data, train_steps = dataset.get_generator(dataset.training, batch_size=args.batch_size, augmentor=augmentor)
    valid_data, valid_steps = dataset.get_generator(dataset.validation, batch_size=args.batch_size, augmentor=None)

    model.fit(epochs=args.epochs,
              training_data=train_data,
              training_steps=train_steps,
              validation_data=valid_data,
              validation_steps=valid_steps,
              plateau_factor=args.plateau_factor,
              plateau_cooldown=args.plateau_cooldown,
              plateau_patience=args.plateau_patience,
              patience=args.patience,
              verbose=1)

    print(model.training_logger)

    test_data, test_steps = dataset.get_generator(dataset.test, batch_size=args.batch_size, augmentor=None)

    predictions, _ = model.predict(test_data=test_data,
                                   test_steps=test_steps,
                                   top_paths=args.top_paths,
                                   beam_width=args.beam_width,
                                   ctc_decode=True,
                                   token_decode=True,
                                   verbose=1)

    baseline_metrics, _ = model.evaluate(dataset.test,
                                         predictions=predictions,
                                         share_top_paths=args.share_top_paths,
                                         prediction_samples=args.prediction_samples,
                                         origin='baseline')

    model.save_context(dataset=dataset, augmentor=augmentor, baseline_metrics=baseline_metrics)

    if args.spell_checker:
        spelling = Spelling(spell_checker=args.spell_checker,
                            api_key=args.api_key,
                            env_key=args.env_key)

        spelling_predictions = spelling.enhance(predictions)

        spelling_metrics, _ = model.evaluate(dataset.test,
                                             predictions=spelling_predictions,
                                             share_top_paths=args.share_top_paths,
                                             prediction_samples=args.prediction_samples,
                                             origin=args.spell_checker)

        model.save_context(spelling_metrics=spelling_metrics)

    print(model.test_logger)
