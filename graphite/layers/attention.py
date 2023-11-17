import tensorflow as tf


class SpectralSelfAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SpectralSelfAttention, self).__init__()
        self.units = units

        self.query_conv = tf.keras.Sequential([
            tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(filters=units // 8, kernel_size=1, padding='same', use_bias=False)),
            tf.keras.layers.ReLU(),
        ])

        self.key_conv = tf.keras.Sequential([
            tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(filters=units // 8, kernel_size=1, padding='same', use_bias=False)),
            tf.keras.layers.ReLU(),
        ])

        self.value_conv = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=units, kernel_size=1, padding='same', use_bias=False))

        self.gamma = tf.Variable(initial_value=tf.zeros(1), trainable=True)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x):
        batchsize, height, width, _ = x.shape

        proj_query = self.query_conv(x)
        proj_query = tf.reshape(proj_query, [batchsize, -1, width * height])
        proj_query = tf.transpose(proj_query, [0, 2, 1])

        proj_key = self.key_conv(x)
        proj_key = tf.reshape(proj_key, [batchsize, -1, width * height])

        energy = tf.matmul(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value = self.value_conv(x)
        proj_value = tf.reshape(proj_value, [batchsize, -1, width * height])

        out = tf.matmul(attention, proj_value)
        out = tf.reshape(out, [batchsize, height, width, self.units])

        return self.gamma * out + x
