import cv2

from dataset import Augmentor, Dataset


def check(args):
    """
    Check and display data samples from a dataset.

    Parameters
    ----------
    args : object
        Command line arguments.
    """

    # def print_section(content):
    #     print(f"\n{'=' * 72}\n{content}\n{'=' * 72}\n")

    dataset = Dataset(source=args.source,
                      level=args.level,
                      shape=args.shape,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      binarization=args.binarization,
                      lazy_mode=args.lazy_mode,
                      seed=args.seed)

    print(dataset)

    # if args.verbose > 0:
    #     print_section(dataset)

    augmentor = Augmentor(erode=args.erode,
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

    # if args.verbose > 1:
    #     print_section(augmentor)

    # src_generator, _ = dataset.get_generator(dataset.training,
    #                                          batch_size=args.batch_size,
    #                                          augmentor=None,
    #                                          raw_data=True,
    #                                          shuffle=False)

    # aug_generator, _ = dataset.get_generator(dataset.training,
    #                                          batch_size=args.batch_size,
    #                                          augmentor=augmentor,
    #                                          raw_data=False,
    #                                          shuffle=False)

    # print("\nChecking samples...\n")

    # while True:
    #     src_images, src_labels = next(src_generator)
    #     aug_images, aug_labels = next(aug_generator)

    #     for i in range(len(src_images)):
    #         cv2.imshow("Source Image", src_images[i])
    #         cv2.imshow("Augmented Image", aug_images[i])

    #         print("\nSource Label")

    #         for j in range(len(src_labels[i])):
    #             print("Length", len(src_labels[i][j]))
    #             print(src_labels[i][j])

    #         print("\nEncoded Label")

    #         for j in range(len(aug_labels[i])):
    #             print("Length", len(aug_labels[i][j]))
    #             print([list(x) for x in aug_labels[i][j]])

    #         print("\n\nPress Enter to continue or Esc to stop...\n")
    #         key = cv2.waitKey(0)

    #         if key == 27:
    #             cv2.destroyAllWindows()
    #             return
