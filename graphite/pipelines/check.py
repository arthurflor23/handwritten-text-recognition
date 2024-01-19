import cv2

from data import Augmentor, Dataset


def check(args):
    """
    Checks and displays samples from the dataset.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object containing all the arguments required.
    """

    dataset = Dataset(source=args.source,
                      text_level=args.text_level,
                      image_shape=args.image_shape,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      lazy_mode=args.lazy_mode,
                      multigrams=True,
                      input_path=args.source_input_path,
                      seed=args.seed)
    print(dataset)

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

    source_gen, _ = dataset.get_generator(data_partition='training',
                                          batch_size=args.batch_size,
                                          batch_encoded=False,
                                          batch_padding=False,
                                          batch_processing=False,
                                          augmentor=None,
                                          shuffle=False)

    encoded_gen, _ = dataset.get_generator(data_partition='training',
                                           batch_size=args.batch_size,
                                           batch_encoded=True,
                                           batch_padding=False,
                                           batch_processing=False,
                                           augmentor=None,
                                           shuffle=False)

    padded_gen, _ = dataset.get_generator(data_partition='training',
                                          batch_size=args.batch_size,
                                          batch_encoded=True,
                                          batch_padding=True,
                                          batch_processing=False,
                                          augmentor=None,
                                          shuffle=False)

    augmented_gen, _ = dataset.get_generator(data_partition='training',
                                             batch_size=args.batch_size,
                                             batch_encoded=True,
                                             batch_padding=True,
                                             batch_processing=False,
                                             augmentor=augmentor,
                                             shuffle=False)

    processed_gen, _ = dataset.get_generator(data_partition='training',
                                             batch_size=args.batch_size,
                                             batch_encoded=True,
                                             batch_padding=True,
                                             batch_processing=True,
                                             augmentor=augmentor,
                                             shuffle=False)

    if args.check:
        print('\nChecking samples...\n')

        while True:
            _, y_source_data = next(source_gen)
            image_source_data, text_source_data, writer_source_data = y_source_data

            _, y_encoded_data = next(encoded_gen)
            image_encoded_data, text_encoded_data, writer_encoded_data = y_encoded_data

            _, y_padded_data = next(padded_gen)
            image_padded_data, text_padded_data, _, = y_padded_data

            x_augmented_data, _ = next(augmented_gen)
            image_augmented_data, _ = x_augmented_data

            x_processed_data, _ = next(processed_gen)
            image_processed_data, text_augmented_data = x_processed_data

            for i in range(len(image_source_data)):
                # source
                print('\n')
                print('Path image')
                print(image_source_data[i], '\n')

                print('Source writer :', writer_source_data[i])
                print('Encoded writer:', writer_encoded_data[i], '\n')

                print('Source text', f"(length {len(text_source_data[i])})")
                print(text_source_data[i])
                print('--------------------------------------------------\n')

                # no augmentation and no padding
                cv2.imshow('Image', image_encoded_data[i])
                print('Encoded text')
                print(text_encoded_data[i])
                print('--------------------------------------------------\n')

                # no augmentation and with padding
                cv2.imshow('Padded image', image_padded_data[i])
                print('Padded text')
                print(text_padded_data[i].tolist())
                print('--------------------------------------------------\n')

                # random text augmented
                print('Text augmented (multigrams)')
                print(dataset.tokenizer.decode_text(text_augmented_data[i].tolist()))
                print()
                print(text_augmented_data[i].tolist())
                print('--------------------------------------------------\n')

                # with augmentation and with padding
                cv2.imshow('Augmented image', image_augmented_data[i])

                # with augmentation, with padding and input process
                cv2.imshow('Processed image', image_processed_data[i])

                print('Press Enter to continue or Esc to stop...\n')
                key = cv2.waitKey(0)

                if key == 27:
                    cv2.destroyAllWindows()
                    return
