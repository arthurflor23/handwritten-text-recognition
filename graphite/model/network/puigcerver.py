import tensorflow as tf

from .layers import InputProcess


class Network():
    """
    A TensorFlow-based class representing `puigcerver` optical model.

    Reference:
        Joan Puigcerver.
        Are multidimensional recurrent layers really necessary for handwritten text recognition?
        14th Document Analysis and Recognition (ICDAR), pp. 67-72, 2017.
        DOI: https://doi.org/10.1109/ICDAR.2017.20
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

        inputs = tf.keras.Input(shape=(None, None, 1))
        inproc = InputProcess(target_shape=self.input_shape)(inputs)

        cnn = tf.keras.layers.Conv2D(filters=16,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same')(inproc)

        cnn = tf.keras.layers.BatchNormalization()(cnn)
        cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)
        cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(cnn)

        cnn = tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same')(cnn)

        cnn = tf.keras.layers.BatchNormalization()(cnn)
        cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)
        cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(cnn)
        cnn = tf.keras.layers.Dropout(rate=0.2)(cnn)

        cnn = tf.keras.layers.Conv2D(filters=48,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same')(cnn)

        cnn = tf.keras.layers.BatchNormalization()(cnn)
        cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)
        cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(cnn)
        cnn = tf.keras.layers.Dropout(rate=0.2)(cnn)

        cnn = tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same')(cnn)

        cnn = tf.keras.layers.BatchNormalization()(cnn)
        cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)
        cnn = tf.keras.layers.Dropout(rate=0.2)(cnn)

        cnn = tf.keras.layers.Conv2D(filters=80,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same')(cnn)

        cnn = tf.keras.layers.BatchNormalization()(cnn)
        cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)

        shape = cnn.get_shape()
        blstm = tf.keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)

        blstm = tf.keras.layers.Dropout(rate=0.5)(blstm)
        dense = tf.keras.layers.Dense(units=output_shape[-1], activation='softmax')(blstm)

        outputs = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(dense)

        return (inputs, outputs)
