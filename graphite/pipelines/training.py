# from data import Augmentor, Dataset
# from models import Model
# from spelling import Spelling

import numpy as np
import tensorflow as tf

from models.components.callback import GANMonitor
from models.synthesis.gan import SynthesisModel


def training(args):
    """
    Performs the training phase.

    Args:
        args (argparse.Namespace):
            A namespace object containing all the arguments required.
    """

    # print(args)

    # Instantiate the model
    model = SynthesisModel(image_shape=[1024, 128, 1],
                           patch_shape=[32, 32, 1],
                           lexical_shape=[1, 144, 150],
                           writer_dim=372,
                           latent_dim=128,
                           embedding_dim=128,
                           backbone_blocks=[16, 32, 64, 128],
                           generator_blocks=[256, 128, 64, 64],
                           discriminator_blocks=[64, 128, 256, 256])

    # model = SynthesisModel(image_shape=[512, 64, 1],
    #                        patch_shape=[32, 32, 1],
    #                        lexical_shape=[1, 144, 150],
    #                        writer_dim=372,
    #                        latent_dim=64,
    #                        embedding_dim=128,
    #                        backbone_blocks=[16, 32, 64, 128],
    #                        generator_blocks=[256, 128, 64, 64],
    #                        discriminator_blocks=[64, 128, 256, 256])

    # model = SynthesisModel(image_shape=[256, 64, 1],
    #                        patch_shape=[32, 32, 1],
    #                        lexical_shape=[1, 32, 80],
    #                        writer_dim=372,
    #                        latent_dim=32,
    #                        embedding_dim=120,
    #                        backbone_blocks=[16, 32, 64, 128],
    #                        generator_blocks=[256, 128, 64, 64],
    #                        discriminator_blocks=[64, 128, 256, 256])

    model.compile(learning_rate=0.001)

    def batch_generator(batch_size):
        # Define your data arrays here
        train_image_data = (np.random.normal(size=[100, 1024, 128, 1]) / 127.5) - 1
        train_aug_image_data = (np.random.normal(size=[100, 1024, 128, 1]) / 127.5) - 1
        # train_text_data = np.random.randint(1, 150, size=[100, 1, 144])
        # train_aug_text_data = np.random.randint(1, 150, size=[100, 1, 144])
        train_writer_data = np.random.randint(0, 372, size=[100])

        vocab_size = 150  # Assuming 150 includes the blank label
        sequence_length = 144
        num_samples = 100

        # Generate random sequences
        train_text_data = np.full((num_samples, 1, sequence_length), vocab_size)  # Fill with blank label
        for i in range(num_samples):
            seq_len = np.random.randint(1, sequence_length)  # Random sequence length
            train_text_data[i, 0, :seq_len] = np.random.randint(1, vocab_size - 2, size=[seq_len])

        train_aug_text_data = train_text_data

        # train_image_data = (np.random.normal(size=[100, 512, 64, 1]) / 127.5) - 1
        # train_aug_image_data = (np.random.normal(size=[100, 512, 64, 1]) / 127.5) - 1
        # train_text_data = np.random.randint(0, 150, size=[100, 1, 144])
        # train_aug_text_data = np.random.randint(0, 150, size=[100, 1, 144])
        # train_writer_data = np.random.randint(0, 372, size=[100])

        # train_image_data = (np.random.normal(size=[100, 256, 64, 1]) / 127.5) - 1
        # train_aug_image_data = (np.random.normal(size=[100, 256, 64, 1]) / 127.5) - 1
        # train_text_data = np.random.randint(0, 80, size=[100, 1, 32])
        # train_aug_text_data = np.random.randint(0, 80, size=[100, 1, 32])
        # train_writer_data = np.random.randint(0, 372, size=[100])

        total_samples = len(train_image_data)
        num_batches = total_samples // batch_size

        for i in range(num_batches):
            # Input data (x)
            batch_aug_image_data = train_aug_image_data[i * batch_size:(i + 1) * batch_size]
            batch_aug_text_data = train_aug_text_data[i * batch_size:(i + 1) * batch_size]

            # Target data (y)
            batch_image_data = train_image_data[i * batch_size:(i + 1) * batch_size]
            batch_text_data = train_text_data[i * batch_size:(i + 1) * batch_size]
            batch_writer_data = train_writer_data[i * batch_size:(i + 1) * batch_size]

            batch_train_data = [batch_image_data,
                                batch_text_data,
                                batch_writer_data,
                                batch_aug_image_data,
                                batch_aug_text_data]

            yield (batch_train_data, [])

    # Create a batch generator
    batch_size = 8
    generator = batch_generator(batch_size)
    train_steps = 100 // batch_size

    model.fit(x=generator,
              steps_per_epoch=train_steps,
              epochs=3,
              callbacks=[
                  #   GANMonitor(filepath='temp',
                  #              latent_dim=128,
                  #              input_data=[
                  #                  (np.random.normal(size=[8, 1024, 128, 1]) / 127.5) - 1, np.full((8, 1, 144), 150)],
                  #              batch_size=4,
                  #              metric='kid'),
                  tf.keras.callbacks.EarlyStopping(monitor='kid',
                                                   min_delta=1e-7,
                                                   patience=1,
                                                   verbose=1,
                                                   mode='min',
                                                   baseline=None,
                                                   restore_best_weights=True,
                                                   start_from_epoch=0),
                  #   tf.keras.callbacks.ModelCheckpoint(
                  #       filepath='temp/model.h5',
                  #       monitor='kid',
                  #       verbose=1,
                  #       save_best_only=True,
                  #       save_weights_only=True,
                  #       mode='min',
                  #       save_freq='epoch',
                  #       options=None,
                  #       initial_value_threshold=None),
                  #   tf.keras.callbacks.ReduceLROnPlateau(
                  #       mode='min',
                  #       monitor='kid',
                  #       min_lr=1e-4,
                  #       min_delta=1e-7,
                  #       factor=0.1,
                  #       cooldown=0,
                  #       patience=20,
                  #       verbose=1),
              ])

    ##########################################
    # # Generate random numpy data as placeholders
    # train_image_data = np.random.normal(size=[100, 1024, 128, 1])
    # train_text_data = np.random.randint(0, 150, size=[100, 1, 144])
    # train_writer_data = np.random.randint(0, 372, size=[100, 1])

    # # Fit the model with numpy arrays
    # data = [train_image_data, train_text_data, train_writer_data]
    # model.fit(data, data, epochs=5, batch_size=8)
    ##########################################

    # # Predict with random data
    # test_image_data = np.random.normal(size=[1, 1024, 128, 1])  # Batch size of 1, image shape of 1024x128x1
    # test_text_data = np.random.randint(0, 150, size=[1, 1, 144])  # Batch size of 1, sequence length of 144
    # # Fill the last few positions with 0s
    # test_text_data[:, -5:] = 0

    # predicted_image = model.generator([test_image_data, test_text_data])
    # print(predicted_image.shape)  # Should match the expected output shape

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
