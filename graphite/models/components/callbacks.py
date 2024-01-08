import os
import cv2
import numpy as np
import tensorflow as tf


class GANMonitor(tf.keras.callbacks.Callback):
    """
    A callback to monitor and save images during GAN training.

    This callback saves generated images at the end of each training step
        where an improvement in the specified metric is observed.
    """

    def __init__(self,
                 filepath,
                 sample_gen,
                 sample_steps,
                 latent_dim,
                 save_freq=200):
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
            Step frequency for saving images.
        """

        self.filepath = filepath
        self.sample_gen = sample_gen
        self.sample_steps = sample_steps
        self.latent_dim = latent_dim
        self.save_freq = save_freq

    def _save_images(self, step, images, name):
        """
        Save a batch of images to the specified filepath.

        Parameters
        ----------
        step : int
            The current step number.
        images : np.ndarray
            Array of images to be saved.
        name : str
            Base name for saved image files.
        """

        filepath = os.path.join(self.filepath, f"step{step + 1}")
        os.makedirs(filepath, exist_ok=True)

        images = np.transpose((images + 1.0) * 127.5, (0, 2, 1, 3))

        for j, image in enumerate(images):
            filename = os.path.join(filepath, f"{j + 1}_{name}.png")
            cv2.imwrite(filename, image)

    def on_train_batch_end(self, step, logs=None):
        """
        Callback function for the end of a training step.

        If an improvement in the specified metric is observed, generates
            and saves random latent, guided latent, and original images.

        Parameters
        ----------
        step : int
            The current step number.
        logs : dict, optional
            Currently available log data.
        """

        if step > 0 and step % self.save_freq == 0:
            for _ in range(self.sample_steps):
                _, sample_data = next(self.sample_gen)
                image_data, text_data, _ = sample_data

                # original images
                self._save_images(step, image_data, name='authentic')

                # guided latent images
                guided_features_inputs, _ = self.model.style_backbone(image_data, training=False)
                guided_latent_inputs, _, _ = self.model.style_encoder(guided_features_inputs, training=False)
                guided_latent_images = self.model.generator([guided_latent_inputs, text_data], training=False)
                self._save_images(step, guided_latent_images, name='guided_style')

                # random latent images
                random_latent_inputs = tf.random.normal(shape=(len(image_data), self.latent_dim))
                random_latent_images = self.model.generator([random_latent_inputs, text_data], training=False)
                self._save_images(step, random_latent_images, name='random_style')
