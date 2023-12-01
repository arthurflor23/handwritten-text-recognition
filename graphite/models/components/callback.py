import os
import tensorflow as tf
import matplotlib.pyplot as plt


class GANMonitor(tf.keras.callbacks.Callback):

    def __init__(self, filepath, latent_dim, num_img=3):

        self.filepath = filepath
        self.latent_dim = latent_dim
        self.num_img = num_img
        os.makedirs(filepath, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):

        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.g_model.predict(random_latent_vectors)

        print('######################')
        print(random_latent_vectors)
        print('######################')
        print(generated_images)
        print('######################')

        for i in range(self.num_img):
            img = generated_images[i]
            img = (img * 127.5) + 127.5
            img = img.astype('uint8')
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(os.path.join(self.filepath, f'generated_img_{epoch}_{i}.png'))
            plt.close()
