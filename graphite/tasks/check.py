import cv2

from dataset import Augmentor, Dataset


def check(args):
    """
    Check and display data samples from a dataset.

    Parameters
    ----------
    args : object
        Command line arguments.

    Returns
    -------
    None
        This function does not return any value.
    """

    dataset = Dataset(source=args.source,
                      level=args.level,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      lazy_mode=True,
                      seed=42)
    print(dataset)

    augmentor = Augmentor(elastic_transform=args.elastic_transform,
                          erosion=args.erosion,
                          dilation=args.dilation,
                          mixup=args.mixup,
                          perspective_transform=args.perspective_transform,
                          gaussian_noise=args.gaussian_noise,
                          gaussian_blur=args.gaussian_blur,
                          shearing=args.shearing,
                          scaling=args.scaling,
                          rotation=args.rotation,
                          translation=args.translation,
                          seed=42)
    print(augmentor)

    src_generator, _ = dataset.get_generator(dataset.training,
                                             batch_size=args.batch_size,
                                             augmentor=None,
                                             raw_data=True,
                                             shuffle=False)

    aug_generator, _ = dataset.get_generator(dataset.training,
                                             batch_size=args.batch_size,
                                             augmentor=augmentor,
                                             raw_data=False,
                                             shuffle=False)

    print("\nChecking samples...\n")

    while True:
        src_images, src_labels = next(src_generator)
        aug_images, aug_labels = next(aug_generator)

        for i in range(len(src_images)):
            cv2.imshow("Source Image", src_images[i])
            cv2.imshow("Augmented Image", aug_images[i])

            print("\nSource Label")

            for j in range(len(src_labels[i])):
                print("Length", len(src_labels[i][j]))
                print(src_labels[i][j])

            print("\nEncoded Label")

            for j in range(len(aug_labels[i])):
                print("Length", len(aug_labels[i][j]))
                print([list(x) for x in aug_labels[i][j]])

            print("\n\nPress Enter to continue or Esc to stop...\n")
            key = cv2.waitKey(0)

            if key == 27:
                cv2.destroyAllWindows()
                return
