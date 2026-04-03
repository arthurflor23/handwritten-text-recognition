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
                      char_width=args.char_width,
                      mask_by_text=args.mask_by_text,
                      order_by_text=args.order_by_text,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      illumination=args.illumination,
                      binarization=args.binarization,
                      lazy_mode=args.lazy_mode,
                      input_path=args.input_path,
                      seed=args.seed)
    print(dataset)

    augmentor = Augmentor(mixup=args.mixup,
                          erode=args.erode,
                          dilate=args.dilate,
                          elastic=args.elastic,
                          perspective=args.perspective,
                          shear=args.shear,
                          rotate=args.rotate,
                          scale=args.scale,
                          shift_y=args.shift_y,
                          shift_x=args.shift_x,
                          salt_and_pepper=args.salt_and_pepper,
                          gaussian_noise=args.gaussian_noise,
                          gaussian_blur=args.gaussian_blur,
                          seed=args.seed)
    print(augmentor)

    src_gen, _ = dataset.get_generator(data_partition='test',
                                       batch_size=args.batch_size,
                                       batch_encoded=False,
                                       batch_processing=False,
                                       batch_scale=False,
                                       augmentor=None,
                                       shuffle=False)

    enc_gen, _ = dataset.get_generator(data_partition='test',
                                       batch_size=args.batch_size,
                                       batch_encoded=True,
                                       batch_processing=False,
                                       batch_scale=False,
                                       augmentor=None,
                                       shuffle=False)

    aug_gen, _ = dataset.get_generator(data_partition='test',
                                       batch_size=args.batch_size,
                                       batch_encoded=True,
                                       batch_processing=False,
                                       batch_scale=False,
                                       augmentor=augmentor,
                                       shuffle=False)

    print('\nChecking samples...')

    while True:
        _, y_src = next(src_gen)
        img_src, txt_src, wri_src, _, _ = y_src

        _, y_enc = next(enc_gen)
        img_enc, txt_enc, wri_enc, _, _ = y_enc

        x_aug, _ = next(aug_gen)
        img_aug, _, _, _, seg_aug = x_aug

        for i in range(len(img_src)):
            print('\nPath image')
            print(img_src[i], '\n')

            print('Writer (src):', wri_src[i])
            print('Writer (enc):', wri_enc[i], '\n')

            print(f'Text (src) length {len(txt_src[i])}')
            print(txt_src[i])
            print('-' * 68, '\n')

            print('Text (enc)')
            print(txt_enc[i])
            print('-' * 68, '\n')

            cv2.imshow('Image (enc)', img_enc[i])
            cv2.imshow('Image (aug)', img_aug[i])
            cv2.imshow('Segmentation (aug)', seg_aug[i])

            print('Press Enter to continue or Esc to stop...\n')
            key = cv2.waitKey(0)

            if key == 27:
                cv2.destroyAllWindows()
                return
