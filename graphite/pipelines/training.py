from data import Augmentor, Dataset
from models import Graphite


def training(args):
    """
    Performs the training phase.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object containing all the arguments required.
    """

    tokenizer, artifacts_path = Graphite().get_tokenizer(synthesis=args.synthesis,
                                                         synthesis_index=args.synthesis_index,
                                                         recognition=args.recognition,
                                                         recognition_index=args.recognition_index,
                                                         experiment_name=args.experiment_name)

    dataset = Dataset(source=args.source,
                      text_level=args.text_level,
                      image_shape=args.image_shape,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      lazy_mode=args.lazy_mode,
                      tokenizer=tokenizer,
                      multigrams=('synthesis' in args.workflow),
                      seed=args.seed)
    print(dataset)

    augmentor = None
    if not args.disable_augmentation:
        augmentor = Augmentor(binarize=args.binarize,
                              erode=args.erode,
                              dilate=args.dilate,
                              elastic=args.elastic,
                              perspective=args.perspective,
                              mixup=args.mixup,
                              shear=args.shear,
                              scale=args.scale,
                              rotate=args.rotate,
                              shift_y=args.shift_y,
                              shift_x=args.shift_x,
                              salt_and_pepper=args.salt_and_pepper,
                              gaussian_noise=args.gaussian_noise,
                              gaussian_blur=args.gaussian_blur,
                              seed=args.seed)
        print(augmentor)

    graphite = Graphite(workflow=args.workflow,
                        synthesis=args.synthesis,
                        recognition=args.recognition,
                        spelling=args.spelling,
                        image_shape=args.image_shape,
                        tokenizer=dataset.tokenizer,
                        synthesis_ratio=args.synthesis_ratio,
                        experiment_name=args.experiment_name)
    print(graphite)

    graphite.compile(learning_rate=args.learning_rate)
    # graphite.compile(learning_rate=args.learning_rate, artifacts_path=artifacts_path)

    training_data, training_steps = dataset.get_generator(partition='training',
                                                          batch_size=args.batch_size,
                                                          augmentor=augmentor,
                                                          shuffle=True)

    validation_data, validation_steps = dataset.get_generator(partition='validation',
                                                              batch_size=args.batch_size,
                                                              augmentor=None,
                                                              shuffle=False)

    monitor_samples_data, monitor_samples_steps = dataset.get_generator(partition='training',
                                                                        samples=args.batch_size,
                                                                        batch_size=args.batch_size,
                                                                        augmentor=None,
                                                                        shuffle=False)

    graphite.fit(epochs=args.epochs,
                 training_data=training_data,
                 training_steps=training_steps,
                 validation_data=validation_data,
                 validation_steps=validation_steps,
                 monitor_samples_data=monitor_samples_data,
                 monitor_samples_steps=monitor_samples_steps,
                 plateau_factor=args.plateau_factor,
                 plateau_cooldown=args.plateau_cooldown,
                 plateau_patience=args.plateau_patience,
                 patience=args.patience)

    test_data, test_steps = dataset.get_generator(partition='test',
                                                  batch_size=args.batch_size,
                                                  augmentor=None,
                                                  shuffle=False)

    # if 'recognition' not in args.workflow:
    #     generations = graphite.generate(test_data=test_data,
    #                                     test_steps=test_steps)

    predictions, _ = graphite.predict(test_data=test_data,
                                      test_steps=test_steps,
                                      top_paths=args.top_paths,
                                      beam_width=args.beam_width,
                                      ctc_decode=True,
                                      token_decode=True)

    exit()

    ##########################################
    # # def print_section(content):
    # #     print(f"\n{'=' * 72}\n{content}\n{'=' * 72}\n")

    # dataset = Dataset(source=args.source,
    #                   text_level=args.text_level,
    #                   image_shape=args.image_shape,
    #                   training_ratio=args.training_ratio,
    #                   validation_ratio=args.validation_ratio,
    #                   test_ratio=args.test_ratio,
    #                   binarization=args.binarization,
    #                   lazy_mode=args.lazy_mode,
    #                   seed=args.seed)

    # if args.verbose > 0:
    #     # print_section(dataset)
    #     print(dataset)

    # augmentor = None
    # if not args.disable_augmentation:
    #     augmentor = Augmentor(erode=args.erode,
    #                           dilate=args.dilate,
    #                           elastic=args.elastic,
    #                           perspective=args.perspective,
    #                           mixup=args.mixup,
    #                           shear=args.shear,
    #                           scale=args.scale,
    #                           rotate=args.rotate,
    #                           shift_y=args.shift_y,
    #                           shift_x=args.shift_x,
    #                           salt_and_pepper=args.salt_and_pepper,
    #                           gaussian_noise=args.gaussian_noise,
    #                           gaussian_blur=args.gaussian_blur,
    #                           seed=args.seed)

    #     if args.verbose > 1:
    #         print_section(augmentor)

    # model = Model(network=args.network, experiment_name=args.experiment_name, seed=args.seed)
    # model.compile(tokenizer=dataset.tokenizer, learning_rate=args.learning_rate, run_index=args.run_index)

    # if args.verbose > 1:
    #     print_section(model)

    # train_data, train_steps = dataset.get_generator(partition=dataset.training,
    #                                                 batch_size=args.batch_size,
    #                                                 augmentor=augmentor)

    # valid_data, valid_steps = dataset.get_generator(partition=dataset.validation,
    #                                                 batch_size=args.batch_size,
    #                                                 augmentor=None)

    # model.fit(epochs=args.epochs,
    #           training_data=train_data,
    #           training_steps=train_steps,
    #           validation_data=valid_data,
    #           validation_steps=valid_steps,
    #           plateau_factor=args.plateau_factor,
    #           plateau_cooldown=args.plateau_cooldown,
    #           plateau_patience=args.plateau_patience,
    #           patience=args.patience,
    #           verbose=args.verbose)

    # if args.verbose > 0:
    #     print_section(model.training_logger)

    # test_data, test_steps = dataset.get_generator(partition=dataset.test,
    #                                               batch_size=args.batch_size,
    #                                               shuffle=False)

    # predictions, _ = model.predict(test_data=test_data,
    #                                test_steps=test_steps,
    #                                top_paths=args.top_paths,
    #                                beam_width=args.beam_width,
    #                                ctc_decode=True,
    #                                token_decode=True,
    #                                verbose=args.verbose)

    # if args.verbose > 0:
    #     print_section(model.test_logger)

    # baseline_metrics = model.evaluate(partition=dataset.test,
    #                                   baseline_predictions=predictions,
    #                                   share_top_paths=args.share_top_paths)

    # model.save_context(dataset=dataset, augmentor=augmentor, baseline_metrics=baseline_metrics)

    # if args.spell_checker:
    #     spelling = Spelling(spell_checker=args.spell_checker,
    #                         api_key=args.api_key,
    #                         env_key=args.env_key)

    #     if args.verbose > 1:
    #         print_section(spelling)

    #     spelling_predictions = spelling.enhance(predictions, verbose=args.verbose)

    #     spelling_metrics = model.evaluate(partition=dataset.test,
    #                                       spelling_predictions=spelling_predictions,
    #                                       share_top_paths=args.share_top_paths)

    #     model.save_context(spelling=spelling, spelling_metrics=spelling_metrics)

    # if args.verbose > 0:
    #     print_section(model.evaluation_logger)
