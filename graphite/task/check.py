import cv2

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
    dataset = Dataset(source=args.source, level=args.level, lazy_mode=True)
    print(dataset)

    # Get batches of original and transformed data for training
    batch_original_data = dataset.next_batch('training', transform=False)
    batch_transform_data = dataset.next_batch('training', transform=True)

    if args.check_samples:
        print("Checking data...\n")

        while True:
            # Get the next batch of original and transformed images and labels
            images, labels = next(batch_original_data)
            images_transform, labels_transform = next(batch_transform_data)

            # Display images
            for i in range(len(images)):
                cv2.imshow("Image", images[i])
                cv2.imshow("Image Transform", images_transform[i])

                print("\nLabel")
                for j in range(len(labels[i])):
                    print(labels[i][j])

                print("\nEncoded Label")
                for j in range(len(labels_transform[i])):
                    print(labels_transform[i][j])

                # Wait for key press
                print("\nPress Enter to continue or Esc to stop...")
                key = cv2.waitKey(0)

                # Stop the looping if Esc is pressed
                if key == 27:
                    cv2.destroyAllWindows()
                    return
