import tensorflow as tf

from .layers import GatedConv2D


class Network():
    """
    A TensorFlow-based class representing Bluche and Messina neural network for handwriting recognition.

    Reference:
        Bluche, T., Messina, R.,
        Gated convolutional recurrent neural networks for multilingual handwriting recognition.
        14th IAPR International Conference on Document Analysis and Recognition (ICDAR), pp. 646-651, 2017.
        URL: https://ieeexplore.ieee.org/document/8270042
    """

    def __init__(self, output_shape):
        """
        Initializes a new instance of the Network class.

        Parameters:
        -----------
        output_shape : tuple
            The shape of the model output (max_rows, max_cols, charset_length).
        """

        self.input_shape = (1024, 128, 1)
        self.output_shape = output_shape

    def compile_model(self, learning_rate, loss_func):
        """
        Build and compile the model.

        Parameters:
        -----------
        learning_rate : float
            The learning rate for the optimizer.
        loss_func : function
            The loss function to be used in the model.
            It's supposed to be a CTC (Connectionist Temporal Classification) loss function.

        Returns:
        --------
        model : tf.keras.Model
            The compiled model.
        """

        inputs, outputs = self._get_architecture(self.output_shape)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss_func)

        return model

    def _get_architecture(self, output_shape):
        """
        Define the architecture of the neural network model.

        Parameters:
        -----------
        output_shape : tuple
            The shape of the output tensor.

        Returns:
        --------
        tuple :
            A tuple containing the input and output tensors of the model.
        """

        inputs = tf.keras.layers.Input(name='input', shape=self.input_shape)
        target_shape = (self.input_shape[0] // 2, self.input_shape[1] // 2, self.input_shape[2] * 4)

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
        dense = tf.keras.layers.Dense(units=output_shape[-1], activation='softmax')(blstm)

        outputs = tf.expand_dims(dense, axis=1)

        return (inputs, outputs)
