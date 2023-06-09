import tensorflow as tf

from .layers import GatedConv2D


class Network():

    def __init__(self):
        print('init... flavor=None, model_uri=None, learning_rate=None, loss=None...')

    def compile_model(self, _, input_size, output_size, learning_rate=None, loss=None):

        inputs, outputs = self._get_architecture(input_size, output_size)

        if learning_rate and loss:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=optimizer, loss=loss)

            # model.load_weights(model_uri)

        return model

    def _get_architecture(self, input_size, output_size):

        inputs = tf.keras.layers.Input(name='input', shape=input_size)

        target_shape = (input_size[0] // 2, input_size[1] // 2, input_size[2] * 4)
        cnn = tf.keras.layers.Reshape(target_shape=target_shape)(inputs)

        cnn = tf.keras.layers.Conv2D(filters=8,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='tanh')(cnn)

        cnn = tf.keras.layers.Conv2D(filters=16,
                                     kernel_size=(2, 4),
                                     strides=(2, 4),
                                     padding='same',
                                     activation='tanh')(cnn)

        cnn = GatedConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(cnn)

        cnn = tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='tanh')(cnn)

        cnn = GatedConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(cnn)

        cnn = tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(2, 4),
                                     strides=(2, 4),
                                     padding='same',
                                     activation='tanh')(cnn)

        cnn = GatedConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(cnn)

        cnn = tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     activation='tanh')(cnn)

        cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='valid')(cnn)

        shape = cnn.get_shape()
        blstm = tf.keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(blstm)
        blstm = tf.keras.layers.Dense(units=128, activation='tanh')(blstm)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(blstm)
        outputs = tf.keras.layers.Dense(units=output_size, activation='softmax')(blstm)

        return (inputs, outputs)
