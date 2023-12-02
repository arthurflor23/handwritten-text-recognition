import os
import cv2
import numpy as np
import tensorflow as tf


class GANMonitor(tf.keras.callbacks.Callback):
    """
    A callback to monitor and save images during GAN training.

    This callback saves generated images at the end of each epoch
        where an improvement in the specified metric is observed.
    """

    def __init__(self, filepath, latent_dim, input_data, batch_size=4, metric='kid'):
        """
        Initializes the GANMonitor callback with specified parameters.

        Args:
            filepath (str):
                Path where images will be saved.
            latent_dim (int):
                Dimensionality of the latent space.
            input_data (list):
                Dataset used for generating images.
            batch_size (int, optional):
                Number of samples per batch of input data.
            metric (str, optional):
                Name of the metric to monitor for improvement.
        """

        self.filepath = filepath
        self.latent_dim = latent_dim
        self.input_data = input_data
        self.batch_size = batch_size
        self.metric = metric
        self.best_metric = np.inf

    def _save_images(self, epoch, images, name):
        """
        Saves a batch of images to the specified filepath.

        Args:
            epoch (int):
                The current epoch number.
            images (ndarray):
                Array of images to be saved.
            name (str):
                Base name for saved image files.
        """

        filepath = os.path.join(self.filepath, str(epoch + 1))
        os.makedirs(filepath, exist_ok=True)

        images = np.transpose((images + 1.0) * 127.5, (0, 2, 1, 3))

        for j, image in enumerate(images):
            filename = os.path.join(filepath, f"{j + 1}_{name}.png")
            cv2.imwrite(filename, image)

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback function for end of an epoch.

        If an improvement in the specified metric is observed,
            generates and saves random latent, guided latent, and original images.

        Args:
            epoch (int):
                The current epoch number.
            logs (dict, optional):
                Currently available log data.
        """

        if logs.get(self.metric, np.inf) <= self.best_metric:
            self.best_metric = logs[self.metric]

            for i in range(0, len(self.input_data), self.batch_size):
                images, texts = self.input_data[i:i + self.batch_size]

                # random latent images
                random_latent_inputs = tf.random.normal(shape=(len(images), self.latent_dim))
                random_latent_images = self.model.generator([random_latent_inputs, texts], training=False)
                self._save_images(epoch, random_latent_images, name='random_style')

                # guided latent images
                guided_features_inputs, _ = self.model.style_backbone(images, training=False)
                guided_latent_inputs, _, _ = self.model.style_encoder(guided_features_inputs, training=False)
                guided_latent_images = self.model.generator([guided_latent_inputs, texts], training=False)
                self._save_images(epoch, guided_latent_images, name='guided_style')

                # original images
                self._save_images(epoch, images, name='authentic')
