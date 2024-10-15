import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


class CSVLogger(tf.keras.callbacks.Callback):
    """
    Callback for logging training metrics to a CSV file.
    """

    def __init__(self,
                 filepath=None,
                 monitor=None,
                 mode='min',
                 separator=','):
        """
        Initialize the callback.

        Parameters
        ----------
        filepath : str, optional
            Path to the CSV file where metrics will be saved.
        monitor : str, optional
            Metric to monitor for checkpoints.
        mode : str, optional
            Mode for monitoring, either 'min', or 'max'.
        separator : str, optional
            Separator for the CSV file.
        """

        super().__init__()

        self.filepath = filepath
        self.mode = mode
        self.monitor = monitor
        self.separator = separator

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.

        Parameters
        ----------
        logs : dict, optional
            Metrics at the start of training.
        """

        self.epoch_index = 0
        self.epochs = []

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of each epoch.

        Parameters
        ----------
        epoch : int
            Index of the current epoch.
        logs : dict, optional
            Metrics at the start of the epoch.
        """

        self.epoch_index += 1

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of each batch.

        Parameters
        ----------
        batch : int
            Index of the current batch.
        logs : dict, optional
            Metrics at the end of the batch.
        """

        self.epochs.append({'epoch': self.epoch_index, **logs})

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.

        Parameters
        ----------
        epoch : int
            Index of the current epoch.
        logs : dict, optional
            Metrics at the end of the epoch.
        """

        self.on_batch_end(None, logs=logs)

        opts = [x for x in dir(self.model) if 'optimizer' in x.lower()]
        opts = [x for x in opts if opts != 'optimizer'] if len(opts) > 1 else opts

        lrs = [getattr(self.model, x, None) for x in opts]
        lrs = [float(tf.keras.backend.get_value(x.learning_rate)) for x in lrs if x]

        self.epochs.append({'epoch': self.epoch_index, **{x: y for x, y in zip(opts, lrs)}})

    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        Parameters
        ----------
        logs : dict, optional
            Metrics at the end of training.
        """

        df = pd.DataFrame(self.epochs)

        df = df.groupby('epoch').mean().reset_index()
        df = df.sort_values(by='epoch').reset_index(drop=True)

        if self.mode and self.monitor:
            df['checkpoint'] = getattr(df[self.monitor], f"cum{self.mode}")()
            df['checkpoint'] = np.where(df['checkpoint'].eq(df['checkpoint'].shift()), 0, df['checkpoint'])
            df['checkpoint'] = df['checkpoint'].astype(bool).replace(False, '').replace(True, '*')

        if self.filepath:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            df.to_csv(self.filepath, sep=self.separator, index=False)

        non_metrics = ['epoch', 'lr', 'learning_rate', 'optimizer', 'checkpoint']
        metrics = [x for x in df.columns if not any(y in x for y in non_metrics)]

        self.history = df[metrics].to_dict(orient='list')
        self.model.history = self


class GANMonitor(tf.keras.callbacks.Callback):
    """
    Callback for monitoring and saving images during GAN training.
    """

    def __init__(self,
                 filepath,
                 sample_gen,
                 sample_steps,
                 style_dim,
                 save_freq=200):
        """
        Initialize the callback.

        Parameters
        ----------
        filepath : str
            Path where images will be saved.
        sample_gen : generator
            Generator yielding sample data batches.
        sample_steps : int
            Number of steps per sample run.
        style_dim : int
            Dimensionality of the style latent space.
        save_freq : int, optional
            Frequency (in steps) to save images.
        """

        self.filepath = filepath
        self.sample_gen = sample_gen
        self.sample_steps = sample_steps
        self.style_dim = style_dim
        self.save_freq = save_freq
        self.global_step = 0

    def _save_images(self, step, images, name):
        """
        Save a batch of images.

        Parameters
        ----------
        step : int
            Current step number.
        images : np.ndarray
            Array of images to be saved.
        name : str
            Base name for the saved image files.
        """

        filepath = os.path.join(self.filepath, str(step))
        os.makedirs(filepath, exist_ok=True)

        images = np.array((images + 1.0) * 127.5, dtype=np.uint8)
        images = images.transpose((0, 2, 1, 3))

        for j, image in enumerate(images):
            cv2.imwrite(os.path.join(filepath, f"{j + 1}_{name}.png"), image)

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of each batch.

        Parameters
        ----------
        batch : int
            Index of the current batch.
        logs : dict, optional
            Log data at the end of the batch.
        """

        if self.global_step > 0 and self.global_step % self.save_freq == 0:
            for _ in range(self.sample_steps):
                _, (image_data, text_data, _, mask_data) = next(self.sample_gen)

                self._save_images(self.global_step, image_data, name='authentic')

                features_data = self.model.style_backbone(image_data, training=False)
                features_data = features_data[0] if isinstance(features_data, list) else features_data

                style_data = self.model.style_encoder(features_data, training=False)
                style_data = style_data[0] if isinstance(style_data, list) else style_data

                fake_guided = self.model.generator([text_data, style_data, mask_data], training=False)
                self._save_images(self.global_step, fake_guided, name='guided')

                random_style_data = tf.random.normal(shape=(len(image_data), self.style_dim))
                fake_random = self.model.generator([text_data, random_style_data, mask_data], training=False)
                self._save_images(self.global_step, fake_random, name='random')

        self.global_step += 1
