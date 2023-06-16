import tensorflow as tf

from .layers import InputProcess, FullGatedConv2D


class Network():
    """
    A TensorFlow-based class representing `flor` optical model.

    Reference:
        Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B.,
        HTR-Flor: A Deep Learning System for Offline Handwritten Text Recognition.
        2020 33rd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), pp. 54-61, 2020.
        DOI: https://doi.org/10.1109/SIBGRAPI51738.2020.00016
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

        cnn = tf.keras.layers.Conv2D(filters=16,
                                     kernel_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same',
                                     kernel_initializer='he_uniform')(inproc)

        cnn = tf.keras.layers.PReLU(shared_axes=[1, 2])(cnn)
        cnn = tf.keras.layers.BatchNormalization(renorm=True)(cnn)

        cnn = FullGatedConv2D(filters=16,
                              kernel_size=(3, 3),
                              padding='same')(cnn)

        cnn = tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     kernel_initializer='he_uniform')(cnn)

        cnn = tf.keras.layers.PReLU(shared_axes=[1, 2])(cnn)
        cnn = tf.keras.layers.BatchNormalization(renorm=True)(cnn)

        cnn = FullGatedConv2D(filters=32,
                              kernel_size=(3, 3),
                              padding='same')(cnn)

        cnn = tf.keras.layers.Conv2D(filters=40,
                                     kernel_size=(2, 4),
                                     strides=(2, 4),
                                     padding='same',
                                     kernel_initializer='he_uniform')(cnn)

        cnn = tf.keras.layers.PReLU(shared_axes=[1, 2])(cnn)
        cnn = tf.keras.layers.BatchNormalization(renorm=True)(cnn)

        cnn = FullGatedConv2D(filters=40,
                              kernel_size=(3, 3),
                              padding='same',
                              kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)

        cnn = tf.keras.layers.Dropout(rate=0.2)(cnn)

        cnn = tf.keras.layers.Conv2D(filters=48,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     kernel_initializer='he_uniform')(cnn)

        cnn = tf.keras.layers.PReLU(shared_axes=[1, 2])(cnn)
        cnn = tf.keras.layers.BatchNormalization(renorm=True)(cnn)

        cnn = FullGatedConv2D(filters=48,
                              kernel_size=(3, 3),
                              padding='same',
                              kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)

        cnn = tf.keras.layers.Dropout(rate=0.2)(cnn)

        cnn = tf.keras.layers.Conv2D(filters=56,
                                     kernel_size=(2, 4),
                                     strides=(2, 4),
                                     padding='same',
                                     kernel_initializer='he_uniform')(cnn)

        cnn = tf.keras.layers.PReLU(shared_axes=[1, 2])(cnn)
        cnn = tf.keras.layers.BatchNormalization(renorm=True)(cnn)

        cnn = FullGatedConv2D(filters=56,
                              kernel_size=(3, 3),
                              padding='same',
                              kernel_constraint=tf.keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)

        cnn = tf.keras.layers.Dropout(rate=0.2)(cnn)

        cnn = tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     kernel_initializer='he_uniform')(cnn)

        cnn = tf.keras.layers.PReLU(shared_axes=[1, 2])(cnn)
        cnn = tf.keras.layers.BatchNormalization(renorm=True)(cnn)

        cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(cnn)

        shape = cnn.get_shape()
        bgru = tf.keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

        bgru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.5))(bgru)
        bgru = tf.keras.layers.Dense(units=256)(bgru)

        bgru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.5))(bgru)
        dense = tf.keras.layers.Dense(units=output_shape[-1], activation='softmax')(bgru)

        outputs = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(dense)

        return (inputs, outputs)
