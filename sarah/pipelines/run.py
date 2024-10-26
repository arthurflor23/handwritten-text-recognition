from data import Augmentor, Dataset
from models import Compose


def run(args):
    """
    Executes the training and testing phase.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object containing all the arguments required.
    """

    tokenizer, run_context = Compose().get_tokenizer(synthesis=args.synthesis,
                                                     synthesis_run_id=args.synthesis_run_id,
                                                     recognition=args.recognition,
                                                     recognition_run_id=args.recognition_run_id,
                                                     experiment_name=args.experiment_name,
                                                     finished_runs=args.finished_runs)

    dataset = Dataset(source=args.source,
                      text_level=args.text_level,
                      image_shape=args.image_shape,
                      pad_value=args.pad_value,
                      char_width=args.char_width,
                      order_by_length=args.order_by_length,
                      training_ratio=(args.training_ratio if args.training else 0.0),
                      validation_ratio=(args.validation_ratio if args.training else 0.0),
                      test_ratio=args.test_ratio,
                      lazy_mode=args.lazy_mode,
                      tokenizer=tokenizer,
                      multigrams=bool(args.synthesis),
                      input_path=args.source_input_path,
                      seed=args.seed)
    print(dataset)

    augmentor = None
    if args.training and not args.skip_augmentation:
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

    compose = Compose(synthesis=args.synthesis,
                      recognition=args.recognition,
                      spelling=args.spelling,
                      image_shape=args.image_shape,
                      tokenizer=dataset.tokenizer,
                      discriminator_steps=args.discriminator_steps,
                      generator_steps=args.generator_steps,
                      synthetic_data_ratio=args.synthetic_data_ratio,
                      synthetic_text_ratio=args.synthetic_text_ratio,
                      synthetic_style_ratio=args.synthetic_style_ratio,
                      experiment_name=args.experiment_name,
                      gpu=args.gpu,
                      seed=args.seed)
    print(compose)

    compose.compile(learning_rate=args.learning_rate, run_context=run_context)

    if args.training:
        compose.save_context(params=args,
                             dataset=dataset,
                             augmentor=augmentor,
                             model=compose.model,
                             new_context=args.training)

        training_gen, training_steps = dataset.get_generator(data_partition='training',
                                                             batch_size=args.batch_size,
                                                             augmentor=augmentor,
                                                             shuffle=True)

        validation_gen, validation_steps = dataset.get_generator(data_partition='validation',
                                                                 batch_size=args.batch_size)

        sample_gen, sample_steps = dataset.get_generator(data_partition='training',
                                                         batch_size=args.batch_size,
                                                         samples=args.batch_size)

        compose.fit(epochs=args.epochs,
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
                    verbose=args.verbose)

    if args.recognition:
        if args.training or args.test:
            test_gen, test_steps = dataset.get_generator(data_partition='test',
                                                         batch_size=args.batch_size)

            predictions, probabilities = compose.predict_recognition(x=test_gen,
                                                                     steps=test_steps,
                                                                     top_paths=args.top_paths,
                                                                     beam_width=args.beam_width,
                                                                     ctc_decode=True,
                                                                     token_decode=True,
                                                                     verbose=args.verbose)

            source_gen, source_steps = dataset.get_generator(data_partition='test',
                                                             batch_size=args.batch_size,
                                                             batch_encoded=False)

            metrics, evaluations = compose.evaluate_recognition(x=predictions,
                                                                y=source_gen,
                                                                steps=source_steps,
                                                                probabilities=probabilities,
                                                                verbose=args.verbose)

            compose.save_context(metrics=metrics, evaluations=evaluations, suffix=None)

            if metrics:
                print('-' * 60)
                print('metrics')
                print(str(metrics).strip('{}').replace("'", '').replace(', ', '\n'))
                print('-' * 60)

        if args.spelling:
            data, predictions, probabilities = compose.get_evaluations()
            dataset = Dataset(data=data)

            source_gen, source_steps = dataset.get_generator(data_partition='test',
                                                             batch_size=args.batch_size,
                                                             batch_encoded=False)

            corrections = compose.predict_spelling(x=predictions, steps=source_steps, verbose=args.verbose)

            metrics, evaluations = compose.evaluate_recognition(x=corrections,
                                                                y=source_gen,
                                                                steps=source_steps,
                                                                probabilities=probabilities,
                                                                verbose=args.verbose)

            compose.save_context(metrics=metrics, evaluations=evaluations, suffix='spelling')

            if metrics:
                print('-' * 60)
                print('spelling metrics')
                print(str(metrics).strip('{}').replace("'", '').replace(', ', '\n'))
                print('-' * 60)

    elif args.synthesis:
        if args.training or args.test:
            test_gen, test_steps = dataset.get_generator(data_partition='test',
                                                         batch_size=args.batch_size)

            predictions = compose.predict_synthesis(x=test_gen, steps=test_steps)

            source_gen, source_steps = dataset.get_generator(data_partition='test',
                                                             batch_size=args.batch_size,
                                                             batch_processing=False)

            metrics, evaluations = compose.evaluate_synthesis(x=predictions,
                                                              y=source_gen,
                                                              steps=source_steps,
                                                              verbose=args.verbose)

            compose.save_context(metrics=metrics, evaluation_images=evaluations)

            if metrics:
                print('-' * 60)
                print('metrics')
                print(str(metrics).strip('{}').replace("'", '').replace(', ', '\n'))
                print('-' * 60)
