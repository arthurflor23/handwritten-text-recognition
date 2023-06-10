import tensorflow as tf

from .layers import GatedConv2D


class Network():

    def compile_model(self, output_shape, learning_rate, ctc_loss_func):

        inputs, outputs = self._get_architecture(output_shape)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=ctc_loss_func)

        return model

    def _get_architecture(self, output_shape):

        input_shape = (1024, 128, 1)
        inputs = tf.keras.layers.Input(name='input', shape=input_shape)

        target_shape = (input_shape[0] // 2, input_shape[1] // 2, input_shape[2] * 4)
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

        print(output_shape)

        shape = cnn.get_shape()
        blstm = tf.keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(blstm)
        blstm = tf.keras.layers.Dense(units=128, activation='tanh')(blstm)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(blstm)
        outputs = tf.keras.layers.Dense(units=output_shape[-1], activation='softmax')(blstm)

        return (inputs, outputs)
