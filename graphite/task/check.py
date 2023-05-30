import cv2

from dataset import Augmentor
from dataset import Dataset


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

    augmentor = Augmentor(
        mixup=args.mixup,
        dilation=args.dilation,
        elastic_distortion=args.elastic_distortion,
        perspective_transform=args.perspective_transform,
        gaussian_noise=args.gaussian_noise,
        gaussian_blur=args.gaussian_blur,
        shearing=args.shearing,
        scaling=args.scaling,
        rotation=args.rotation,
        translate_x=args.translate_x,
        translate_y=args.translate_y,
    )
    print(augmentor)

    src_batch = dataset.batch_generator('training', normalize=False, debug=True)
    aug_batch = dataset.batch_generator('training', augmentor=augmentor)

    if args.check_samples:
        print("Checking samples...\n")

        # import time
        # counter = 0
        # start_time = time.time()
        # for _ in range(1000):
        #     aug_images, aug_labels = next(aug_batch)
        #     counter += 1
        # end_time = time.time()

        # print("\n\nLoop performed {} times".format(counter))
        # print(f"Execution time: {end_time - start_time:.4f} seconds\n\n")
        # exit()

        while True:
            src_images, src_labels = next(src_batch)
            aug_images, aug_labels = next(aug_batch)

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
                    print(aug_labels[i][j])

                print("\nPress Enter to continue or Esc to stop...\n")
                key = cv2.waitKey(0)

                if key == 27:
                    cv2.destroyAllWindows()
                    return
