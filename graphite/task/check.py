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

    # Create a dataset object with the specified arguments
    dataset = Dataset(source=args.source,
                      level=args.level,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      lazy_mode=True,
                      seed=42)
    print(dataset)

    # Create Augmentor instance
    augmentor = Augmentor(
        elastic_distortion=args.elastic_distortion,
        perspective_transform=args.perspective_transform,
        gaussian_noise=args.gaussian_noise,
        gaussian_blur=args.gaussian_blur,
        shearing=args.shearing,
        scaling=args.scaling,
        rotation=args.rotation,
        translate_x=args.translate_x,
        translate_y=args.translate_y,
        mixup=args.mixup,
    )
    print(augmentor)

    # Get batches of original and transformed data for training
    src_batch = dataset.batch_generator('training', keep_original=True)
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
            # Get the next batch of original and transformed images and labels
            src_images, src_labels = next(src_batch)
            aug_images, aug_labels = next(aug_batch)

            # Display images
            for i in range(len(src_images)):
                cv2.imshow("Source Image", src_images[i])
                cv2.imshow("Augmented Image", aug_images[i])

                print("\nLabel")
                for j in range(len(src_labels[i])):
                    print(src_labels[i][j])

                print("\nEncoded Label")
                for j in range(len(aug_labels[i])):
                    print(aug_labels[i][j])

                # Wait for key press
                print("\nPress Enter to continue or Esc to stop...")
                key = cv2.waitKey(0)

                # Stop the looping if Esc is pressed
                if key == 27:
                    cv2.destroyAllWindows()
                    return
