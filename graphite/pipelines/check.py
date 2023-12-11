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

    dataset = Dataset(mode='recognition',
                      source=args.source,
                      text_level=args.text_level,
                      image_shape=args.image_shape,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      lazy_mode=args.lazy_mode,
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

    src_generator, _ = dataset.get_generator(partition='training',
                                             batch_size=args.batch_size,
                                             augmentor=None,
                                             use_source=True,
                                             prepare_batch=False,
                                             shuffle=False)

    enc_generator, _ = dataset.get_generator(partition='training',
                                             batch_size=args.batch_size,
                                             augmentor=None,
                                             use_source=False,
                                             prepare_batch=False,
                                             shuffle=False)

    aug_generator, _ = dataset.get_generator(partition='training',
                                             batch_size=args.batch_size,
                                             augmentor=augmentor,
                                             use_source=False,
                                             prepare_batch=False,
                                             shuffle=False)

    inp_generator, _ = dataset.get_generator(partition='training',
                                             batch_size=args.batch_size,
                                             augmentor=augmentor,
                                             use_source=False,
                                             prepare_batch=True,
                                             shuffle=False)

    print('\nChecking samples...\n')
    while True:
        src_images, src_labels = next(src_generator)
        enc_images, enc_labels = next(enc_generator)
        aug_images, _ = next(aug_generator)
        inp_images, inp_labels = next(inp_generator)

        for i in range(args.batch_size):
            # raw image and text
            print('Source Image (path)')
            print(src_images[i], '\n')
            print('Source Label')
            print(src_labels[i], '\n')

            # image and text (no augmentation, no padding)
            cv2.imshow('Image', enc_images[i])
            print('Encoded Label')
            print(enc_labels[i], '\n')

            # image (with augmentation, no padding)
            cv2.imshow('Augmented Image', aug_images[i])

            # image and text (with augmentation, with padding)
            cv2.imshow('Input Image', inp_images[i])
            print('Input Label')
            print(inp_labels[i].squeeze(axis=-1).tolist(), '\n')

            print('Press Enter to continue or Esc to stop...\n\n')
            key = cv2.waitKey(0)

            if key == 27:
                cv2.destroyAllWindows()
                return
