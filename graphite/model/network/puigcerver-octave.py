import tensorflow as tf

from .layers import InputProcess, OctConv2D


class Network():
    """
    A TensorFlow-based class representing `puigcerver-octave` optical model (by @khinggan).

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

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
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

        high, low = OctConv2D(filters=16, alpha=0.25)([inproc, tf.keras.layers.AveragePooling2D(2)(inproc)])

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.01)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.01)(low)
        high = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(high)
        low = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(low)

        high, low = OctConv2D(filters=32, alpha=0.25)([high, low])

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.01)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.01)(low)
        high = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(high)
        low = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(low)

        high = tf.keras.layers.Dropout(rate=0.2)(high)
        low = tf.keras.layers.Dropout(rate=0.2)(low)

        high = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same')(high)
        low = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='same')(low)

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.01)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.01)(low)
        high = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(high)
        low = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(low)

        high = tf.keras.layers.Dropout(rate=0.2)(high)
        low = tf.keras.layers.Dropout(rate=0.2)(low)

        high = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(high)
        low = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(low)

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.01)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.01)(low)

        high = tf.keras.layers.Dropout(rate=0.2)(high)
        low = tf.keras.layers.Dropout(rate=0.2)(low)

        high = tf.keras.layers.Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding='same')(high)
        low = tf.keras.layers.Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding='same')(low)

        high = tf.keras.layers.BatchNormalization()(high)
        low = tf.keras.layers.BatchNormalization()(low)
        high = tf.keras.layers.LeakyReLU(alpha=0.01)(high)
        low = tf.keras.layers.LeakyReLU(alpha=0.01)(low)

        high, low = OctConv2D(filters=80, alpha=0.25)([high, low])

        high = tf.keras.layers.BatchNormalization()(high)
        high = tf.keras.layers.Activation('relu')(high)

        low = tf.keras.layers.BatchNormalization()(low)
        low = tf.keras.layers.Activation('relu')(low)

        high_to_high = tf.keras.layers.Conv2D(80, 3, padding='same')(high)
        low_to_high = tf.keras.layers.Conv2D(80, 3, padding='same')(low)

        low_to_high = tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, 2, axis=1))(low_to_high)
        low_to_high = tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, 2, axis=2))(low_to_high)

        octconv = tf.keras.layers.Add()([high_to_high, low_to_high])
        octconv = tf.keras.layers.BatchNormalization()(octconv)
        octconv = tf.keras.layers.Activation('relu')(octconv)

        shape = octconv.get_shape()
        blstm = tf.keras.layers.Reshape((shape[1], shape[2] * shape[3]))(octconv)

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)
        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.5))(blstm)

        blstm = tf.keras.layers.Dropout(rate=0.5)(blstm)
        dense = tf.keras.layers.Dense(units=output_shape[-1], activation='softmax')(blstm)

        outputs = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(dense)

        return (inputs, outputs)
