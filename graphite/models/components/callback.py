import os
import cv2
import numpy as np
import tensorflow as tf


class GANMonitor(tf.keras.callbacks.Callback):

    def __init__(self, filepath, text_inputs, latent_dim, num_img=3):

        self.filepath = filepath
        self.text_inputs = text_inputs[:num_img]
        self.latent_dim = latent_dim
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):

        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))

        # features_inputs, _ = self.model.b_model.predict(random_latent_vectors)
        # latent_inputs, _, _ = self.model.e_model.predict(features_inputs)

        generated_images = self.model.g_model([random_latent_vectors, self.text_inputs], training=False)
        # generated_images = self.model.g_model.predict([latent_inputs, self.text_inputs])
        generated_images = (generated_images + 1.0) * 127.5
        generated_images = np.transpose(generated_images, (0, 2, 1, 3))

        os.makedirs(self.filepath, exist_ok=True)

        for i in range(self.num_img):
            filepath = os.path.join(self.filepath, f"img_{i}_{epoch}.png")
            image = generated_images[i].squeeze().astype('uint8')

            cv2.imwrite(filepath, image)
