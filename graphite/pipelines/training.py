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
                                                             batch_size=args.batch_size)

    sample_gen, sample_steps = dataset.get_generator(partition='training',
                                                     samples=args.batch_size)

    graphite.fit(epochs=args.epochs,
                 training_gen=training_gen,
                 training_steps=training_steps,
                 validation_gen=validation_gen,
                 validation_steps=validation_steps,
                 monitor_sample_gen=sample_gen,
                 monitor_sample_steps=sample_steps,
                 plateau_factor=args.plateau_factor,
                 plateau_cooldown=args.plateau_cooldown,
                 plateau_patience=args.plateau_patience,
                 patience=args.patience)

    graphite.save_context(params=args,
                          dataset=dataset,
                          augmentor=augmentor,
                          model=graphite.model)

    if 'recognition' in args.workflow:
        test_gen, test_steps = dataset.get_generator(partition='test', batch_size=args.batch_size)
        predictions, _ = graphite.predict_recognition(x=test_gen,
                                                      steps=test_steps,
                                                      top_paths=args.top_paths,
                                                      beam_width=args.beam_width,
                                                      corrections=False)

        test_gen, test_steps = dataset.get_generator(partition='test', batch_size=args.batch_size)
        corrections, _ = graphite.predict_recognition(x=test_gen,
                                                      steps=test_steps,
                                                      top_paths=args.top_paths,
                                                      beam_width=args.beam_width,
                                                      corrections=True)

        source_gen, source_steps = dataset.get_generator(partition='test',
                                                         batch_size=args.batch_size,
                                                         use_source=True)
        p_metrics, p_evaluations = graphite.evaluate_recognition(x=predictions,
                                                                 y=source_gen,
                                                                 steps=source_steps)

        source_gen, source_steps = dataset.get_generator(partition='test',
                                                         batch_size=args.batch_size,
                                                         use_source=True)
        s_metrics, s_evaluations = graphite.evaluate_recognition(x=corrections,
                                                                 y=source_gen,
                                                                 steps=source_steps)

        graphite.save_context(metrics=p_metrics, evaluations=p_evaluations, suffix='prediction')
        graphite.save_context(metrics=s_metrics, evaluations=s_evaluations, suffix='spelling')

    elif 'synthesis' in args.workflow:
        test_gen, test_steps = dataset.get_generator(partition='test', batch_size=args.batch_size)
        predictions = graphite.predict_synthesis(x=test_gen, steps=test_steps)

        test_gen, test_steps = dataset.get_generator(partition='test', batch_size=args.batch_size)
        metrics, evaluations = graphite.evaluate_synthesis(x=predictions, y=test_gen, steps=test_steps)

        graphite.save_context(metrics=metrics, evaluation_images=evaluations, suffix='prediction')
