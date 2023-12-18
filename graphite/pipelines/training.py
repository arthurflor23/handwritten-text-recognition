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

    tokenizer, mlrun = Graphite().get_tokenizer(synthesis=args.synthesis,
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

    graphite.compile(learning_rate=args.learning_rate, mlrun=mlrun)

    training_gen, training_steps = dataset.get_generator(partition='training',
                                                         batch_size=args.batch_size,
                                                         augmentor=augmentor,
                                                         shuffle=True)

    validation_gen, validation_steps = dataset.get_generator(partition='validation',
                                                             batch_size=args.batch_size,
                                                             augmentor=None,
                                                             shuffle=False)

    monitor_samples_gen, monitor_samples_steps = dataset.get_generator(partition='training',
                                                                       samples=args.batch_size,
                                                                       batch_size=args.batch_size,
                                                                       augmentor=None,
                                                                       shuffle=False)

    graphite.fit(epochs=args.epochs,
                 training_gen=training_gen,
                 training_steps=training_steps,
                 validation_gen=validation_gen,
                 validation_steps=validation_steps,
                 monitor_samples_gen=monitor_samples_gen,
                 monitor_samples_steps=monitor_samples_steps,
                 plateau_factor=args.plateau_factor,
                 plateau_cooldown=args.plateau_cooldown,
                 plateau_patience=args.plateau_patience,
                 patience=args.patience)

    graphite.save_context(params=args,
                          dataset=dataset,
                          augmentor=augmentor)

    if 'recognition' in args.workflow:
        test_gen, test_steps = dataset.get_generator(partition='test',
                                                     batch_size=args.batch_size,
                                                     augmentor=None,
                                                     shuffle=False)

        predictions, spelling_predictions, _ = graphite.predict(test_gen=test_gen,
                                                                test_steps=test_steps,
                                                                top_paths=args.top_paths,
                                                                beam_width=args.beam_width,
                                                                ctc_decode=True,
                                                                token_decode=True)

        label_gen, label_steps = dataset.get_generator(partition='test',
                                                       batch_size=args.batch_size,
                                                       augmentor=None,
                                                       use_source=True,
                                                       shuffle=False)

        metrics, evaluations = graphite.evaluate(label_gen=label_gen,
                                                 label_steps=label_steps,
                                                 predictions=predictions)

        spelling_metrics, spelling_evaluations = graphite.evaluate(label_gen=label_gen,
                                                                   label_steps=label_steps,
                                                                   predictions=spelling_predictions)

        graphite.save_context(metrics=metrics,
                              evaluations=evaluations,
                              spelling_metrics=spelling_metrics,
                              spelling_evaluations=spelling_evaluations)
