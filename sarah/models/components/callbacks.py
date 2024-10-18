import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


class GANMonitor(tf.keras.callbacks.Callback):
    """
    Callback for monitoring and saving images during GAN training.
    """

    def __init__(self,
                 filepath,
                 sample_gen,
                 sample_steps,
                 latent_dim,
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
        latent_dim : int
            Dimension of the style latent space.
        save_freq : int, optional
            Frequency (in steps) to save images.
        """

        self.filepath = filepath
        self.sample_gen = sample_gen
        self.sample_steps = sample_steps
        self.latent_dim = latent_dim
        self.save_freq = save_freq

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training.

        Parameters
        ----------
        logs : dict, optional
            Metrics at the start of training.
        """

        self.epoch_index = 0
        self.global_step = 0
        self.local_step = 0

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
        self.local_step = 0

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
            subpath = f"{str(self.global_step)}_{str(self.local_step)}_{str(self.epoch_index)}"
            filepath = os.path.join(self.filepath, subpath)

            for _ in range(self.sample_steps):
                _, (image_data, text_data, _, mask_data) = next(self.sample_gen)

                self._save_images(filepath, image_data, name='authentic')

                features_data = self.model.style_backbone(image_data, training=False)
                features_data = features_data[0] if isinstance(features_data, list) else features_data

                latent_data = self.model.style_encoder([features_data, mask_data], training=False)
                latent_data = latent_data[0] if isinstance(latent_data, list) else latent_data

                fake_guided = self.model.generator([text_data, latent_data], training=False)
                self._save_images(filepath, fake_guided, name='guided')

                random_latent_data = tf.random.normal(shape=(len(image_data), self.latent_dim))
                fake_random = self.model.generator([text_data, random_latent_data], training=False)
                self._save_images(filepath, fake_random, name='random')

        self.global_step += 1
        self.local_step += 1

    def _save_images(self, filepath, images, name):
        """
        Save a batch of images.

        Parameters
        ----------
        filepath : str
            Path where images will be saved.
        images : np.ndarray
            Array of images to be saved.
        name : str
            Base name for the saved image files.
        """

        os.makedirs(filepath, exist_ok=True)

        images = np.array((images + 1.0) * 127.5, dtype=np.uint8)
        images = images.transpose((0, 2, 1, 3))

        for j, image in enumerate(images):
            cv2.imwrite(os.path.join(filepath, f"{j + 1}_{name}.png"), image)


class TrainingLogger(tf.keras.callbacks.Callback):
    """
    Logs training metrics to a CSV file and saves model checkpoints.

    References
    ----------
    Issue: Mismatch Between Training Progress and History/CSVLogger Callback Values
        https://github.com/keras-team/keras/issues/20212
    """

    def __init__(self,
                 mode='min',
                 monitor=None,
                 model_path=None,
                 save_best_only=True,
                 save_weights_only=True,
                 csv_path=None,
                 csv_separator=',',
                 verbose=1):
        """
        Initializes the logger callback.

        Parameters
        ----------
        monitor : str, optional
            Metric to monitor for saving checkpoints.
        mode : str, optional
            Mode for monitoring ('min' or 'max').
        model_path : str, optional
            Path for saving model checkpoints.
        save_best_only : bool, optional
            Whether to save only the best models.
        save_weights_only : bool, optional
            Whether to save only the model's weights.
        csv_path : str, optional
            Path to the CSV file for logging metrics.
        csv_separator : str, optional
            csv_separator for the CSV file.
        verbose : int, optional
            Verbosity mode.
        """

        super().__init__()

        self.mode = mode
        self.monitor = monitor
        self.model_path = model_path
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.csv_path = csv_path
        self.csv_separator = csv_separator
        self.verbose = verbose

        if self.model_path:
            suffix = '.weights.h5' if self.save_weights_only else '.keras'
            self.model_path = f"{self.model_path.rstrip(suffix)}{suffix}"

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
        self.best = float('-inf') if self.mode == 'max' else float('inf')

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

        if self.model_path:
            if self.save_best_only and self.mode and self.monitor in logs.keys():
                df = self._get_current_dataframe()
                current = df[self.monitor].iloc[-1]

                if current < self.best:
                    if self.verbose > 0:
                        print(f"\nEpoch {self.epoch_index}: {self.monitor} improved "
                              f"from {self.best:.5f} to {current:.5f}, "
                              f"saving model to {self.model_path}")

                    if self.save_weights_only:
                        self.model.save_weights(self.model_path, overwrite=True)
                    else:
                        self.model.save(self.model_path, overwrite=True)

                    self.best = current
                else:
                    if self.verbose > 0:
                        print(f"\nEpoch {self.epoch_index}: "
                              f"{self.monitor} did not improve "
                              f"from {self.best:.5f}")
            else:
                if self.verbose > 0:
                    print(f"\nEpoch {self.epoch_index}: saving model to {self.model_path}")

                if self.save_weights_only:
                    self.model.save_weights(self.model_path, overwrite=True)
                else:
                    self.model.save(self.model_path, overwrite=True)

    def on_train_end(self, logs=None):
        """
        Called at the end of training.

        Parameters
        ----------
        logs : dict, optional
            Metrics at the end of training.
        """

        df = self._get_current_dataframe()

        if self.mode and self.monitor in logs.keys():
            df['checkpoint'] = getattr(df[self.monitor], f"cum{self.mode}")()
            df['checkpoint'] = np.where(df['checkpoint'].eq(df['checkpoint'].shift()), 0, df['checkpoint'])
            df['checkpoint'] = df['checkpoint'].astype(bool).replace(False, '').replace(True, '*')

        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            df.to_csv(self.csv_path, sep=self.csv_separator, index=False)

        non_metrics = ['epoch', 'lr', 'learning_rate', 'optimizer', 'checkpoint']
        metrics = [x for x in df.columns if not any(y in x for y in non_metrics)]

        self.history = df[metrics].to_dict(orient='list')
        self.model.history = self

    def _get_current_dataframe(self):
        """
        Retrieves the current training metrics as a DataFrame.
        """

        df = pd.DataFrame(self.epochs)
        df = df.groupby('epoch').mean().reset_index()
        df = df.sort_values(by='epoch').reset_index(drop=True)

        return df
