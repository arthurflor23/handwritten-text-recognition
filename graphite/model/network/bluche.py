import tensorflow as tf

from .layers import GatedConv2D


class Network():

    def compile_model(self,
                      output_size,
                      learning_rate,
                      ctc_loss_func):

        inputs, outputs = self._get_architecture(input_size=(1024, 128, 1), output_size=output_size)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=ctc_loss_func)

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
