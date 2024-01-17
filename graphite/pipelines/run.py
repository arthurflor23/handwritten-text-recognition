from data import Augmentor, Dataset
from models import Graphite


def run(args, training=None):
    """
    Executes the training and testing phase.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object containing all the arguments required.
    training : bool, optional
        Whether to execute training phase.
    """

    tokenizer, run_context = Graphite().get_tokenizer(synthesis=args.synthesis,
                                                      synthesis_run_index=args.synthesis_run_index,
                                                      recognition=args.recognition,
                                                      recognition_run_index=args.recognition_run_index,
                                                      experiment_name=args.experiment_name)

    dataset = Dataset(source=args.source,
                      text_level=args.text_level,
                      image_shape=args.image_shape,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      lazy_mode=args.lazy_mode,
                      tokenizer=tokenizer,
                      multigrams=bool(args.synthesis),
                      input_path=args.source_input_path,
                      seed=args.seed)
    print(dataset)

    augmentor = None
    if training and not args.disable_augmentation:
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

    graphite = Graphite(synthesis=args.synthesis,
                        recognition=args.recognition,
                        spelling=args.spelling,
                        image_shape=dataset.image_shape,
                        tokenizer=dataset.tokenizer,
                        discriminator_steps=args.discriminator_steps,
                        generator_steps=args.generator_steps,
                        synthesis_ratio=args.synthesis_ratio,
                        experiment_name=args.experiment_name)
    print(graphite)

    graphite.compile(learning_rate=args.learning_rate, run_context=run_context)

    if training:
        training_gen, training_steps = dataset.get_generator(data_partition='training',
                                                             batch_size=args.batch_size,
                                                             augmentor=augmentor,
                                                             shuffle=True)

        validation_gen, validation_steps = dataset.get_generator(data_partition='validation',
                                                                 batch_size=args.batch_size)

        sample_gen, sample_steps = dataset.get_generator(data_partition='training',
                                                         batch_size=args.batch_size,
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
                     patience=args.patience,
                     verbose=1)

        graphite.save_context(params=args,
                              dataset=dataset,
                              augmentor=augmentor,
                              model=graphite.model)

    if args.recognition:
        prediction_configs = [{
            'predict': True,
            'corrections': False,
            'suffix': None,
        }, {
            'predict': args.spelling,
            'corrections': True,
            'suffix': 'spelling',
        }]

        for config in prediction_configs:
            if not config['predict']:
                continue

            test_gen, test_steps = dataset.get_generator(data_partition='test',
                                                         batch_size=args.batch_size)

            predictions, probabilities = graphite.predict_recognition(x=test_gen,
                                                                      steps=test_steps,
                                                                      top_paths=args.top_paths,
                                                                      beam_width=args.beam_width,
                                                                      ctc_decode=True,
                                                                      token_decode=True,
                                                                      corrections=config['corrections'],
                                                                      verbose=1)

            source_gen, source_steps = dataset.get_generator(data_partition='test',
                                                             batch_size=args.batch_size,
                                                             batch_encoded=False)

            metrics, evaluations = graphite.evaluate_recognition(x=predictions,
                                                                 y=source_gen,
                                                                 steps=source_steps,
                                                                 probabilities=probabilities,
                                                                 verbose=1)

            graphite.save_context(metrics=metrics, evaluations=evaluations, suffix=config['suffix'])

            if metrics:
                print('--------------------------------------------------\n')
                print(f"{config['suffix'] or ''} metrics".strip(), '\n')
                print(str(metrics).strip('{}').replace("'", '').replace(', ', '\n'))
                print('\n--------------------------------------------------')

    elif args.synthesis:
        test_gen, test_steps = dataset.get_generator(data_partition='test',
                                                     batch_size=args.batch_size)

        predictions = graphite.predict_synthesis(x=test_gen, steps=test_steps)

        source_gen, source_steps = dataset.get_generator(data_partition='test',
                                                         batch_size=args.batch_size,
                                                         batch_processing=False)

        metrics, evaluations = graphite.evaluate_synthesis(x=predictions,
                                                           y=source_gen,
                                                           steps=source_steps,
                                                           verbose=1)

        graphite.save_context(metrics=metrics, evaluation_images=evaluations)

        if metrics:
            print('--------------------------------------------------\n')
            print('metrics', '\n')
            print(str(metrics).strip('{}').replace("'", '').replace(', ', '\n'))
            print('\n--------------------------------------------------')
