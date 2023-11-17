import tensorflow as tf


class ConditionalBatchNormalization(tf.keras.layers.Layer):

    def __init__(self, units, momentum=0.1, epsilon=1e-5):
        super(ConditionalBatchNormalization, self).__init__()

        self.units = units

        self.gain = tf.keras.layers.SpectralNormalization(tf.keras.layers.Dense(units=units, use_bias=False))
        self.bias = tf.keras.layers.SpectralNormalization(tf.keras.layers.Dense(units=units, use_bias=False))
        self.norm = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

    def call(self, inputs, conditional_inputs, training=True):
        gain = 1 + self.gain(conditional_inputs)
        bias = self.bias(conditional_inputs)

        gain = tf.reshape(gain, (-1, 1, 1, self.units))
        bias = tf.reshape(bias, (-1, 1, 1, self.units))

        out = self.norm(inputs, training=training)

        return gain * out + bias

    def compute_output_shape(self, input_shape):
        return input_shape
