from dataset import Augmentor, Dataset
# from model import Model
# from spelling import Spelling

import numpy as np
from model.synthesis.higanplus import SynthesisModel


def training(args):
    """
    Performs training and testing phases.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object that contains all the arguments required.
    """

    # print(args)

    # Instantiate the model
    model = SynthesisModel(vocab_size=80, embedding_dim=120, style_dim=32)
    model.compile(learning_rate=0.001)

    # Generate random numpy data as placeholders
    train_image_data = np.random.normal(size=[100, 512, 64, 1])  # 100 samples, image shape of 512x64x1
    train_text_data = np.random.randint(0, 80, size=[100, 15])  # 100 samples, sequence length of 15
    # Fill the last few positions with 0s
    train_text_data[:, -5:] = 0

    # Fit the model with numpy arrays
    model.fit([train_image_data, train_text_data], train_text_data, epochs=5, batch_size=32)

    # Predict with random data
    test_image_data = np.random.normal(size=[1, 512, 64, 1])  # Batch size of 1, image shape of 512x64x1
    test_text_data = np.random.randint(0, 80, size=[1, 15])  # Batch size of 1, sequence length of 15
    # Fill the last few positions with 0s
    test_text_data[:, -5:] = 0

    predicted_image = model.generator(test_image_data, test_text_data)

    print(predicted_image.shape)  # Should match the expected output shape

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
    ##########################################

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
