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

    def __init__(self,
                 filepath,
                 sample_gen,
                 sample_steps,
                 latent_dim,
                 save_freq=5):
        """
        Initialize the GANMonitor callback with specified parameters.

        Parameters
        ----------
        filepath : str
            Path where images will be saved.
        sample_gen : generator, optional
            Generator yielding samples data batches.
        sample_steps : int, optional
            Number of steps per sample run.
        latent_dim : int
            Dimensionality of the latent space.
        save_freq : int, optional
            Epoch frequency for saving images.
        """

        self.filepath = filepath
        self.sample_gen = sample_gen
        self.sample_steps = sample_steps
        self.latent_dim = latent_dim
        self.save_freq = save_freq

    def _save_images(self, epoch, images, name):
        """
        Save a batch of images to the specified filepath.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        images : np.ndarray
            Array of images to be saved.
        name : str
            Base name for saved image files.
        """

        filepath = os.path.join(self.filepath, f"epoch{epoch + 1}")
        os.makedirs(filepath, exist_ok=True)

        images = np.transpose((images + 1.0) * 127.5, (0, 2, 1, 3))

        for j, image in enumerate(images):
            filename = os.path.join(filepath, f"{j + 1:03}_{name}.png")
            cv2.imwrite(filename, image)

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback function for the end of an epoch.

        If an improvement in the specified metric is observed, generates and saves
            random latent, guided latent, and original images.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : dict, optional
            Currently available log data.
        """

        if (epoch + 1) % self.save_freq == 0:
            for _ in range(self.sample_steps):
                images, texts = next(self.sample_gen)

                # original images
                self._save_images(epoch, images, name='authentic')

                # guided latent images
                guided_features_inputs, _ = self.model.style_backbone(images, training=False)
                guided_latent_inputs, _, _ = self.model.style_encoder(guided_features_inputs, training=False)
                guided_latent_images = self.model.generator([guided_latent_inputs, texts], training=False)
                self._save_images(epoch, guided_latent_images, name='guided_style')

                # random latent images
                random_latent_inputs = tf.random.normal(shape=(len(images), self.latent_dim))
                random_latent_images = self.model.generator([random_latent_inputs, texts], training=False)
                self._save_images(epoch, random_latent_images, name='random_style')
