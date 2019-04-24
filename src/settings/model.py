"""Create the HTR deep learning model using tf.keras"""

import tensorflow as tf
# import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
INPUT_SIZE = (800, 64, 1)
BATCH_SIZE = 50


class HTR():

    def __init__(self):
        labels = tf.keras.Input(name='the_label', shape=(None,), dtype='int32')
        label_length = tf.keras.Input(name='label_length', shape=[1], dtype='int32')

        input_data = tf.keras.Input(name='the_input', shape=INPUT_SIZE, dtype='float32')
        input_length = tf.keras.Input(name='input_length', shape=[1], dtype='int32')

        self.__setup_cnn(input_data)
        self.__setup_rnn(self.cnn_out)
        self.__setup_ctc([self.rnn_out, labels, input_length, label_length])

        self.model = tf.keras.Model(name="HTR", inputs=[input_data, labels, input_length, label_length], outputs=self.ctc_out)
        
        # dummy loss-function for compiling model, actual CTC loss-function defined as a lambda layer
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss={'ctc': lambda y_true, y_pred: y_pred})

    def __setup_cnn(self, inputs):
        """CNN model"""

        # first layer: Conv (5x5) + Pool (2x2) - Output size: 400 x 32 x 64
        init = tf.random_normal_initializer(stddev=0.1)
        conv1 = tf.keras.layers.Conv2D(name="cnn_conv1", filters=64, kernel_size=5, strides=(1, 1), padding="same", activation="relu", kernel_initializer=init)(inputs)
        pool1 = tf.keras.layers.MaxPooling2D(name="cnn_max_pooling1", pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)

        # second layer: Conv (5x5) - Output size: 400 x 32 x 128
        conv2 = tf.keras.layers.Conv2D(name="cnn_conv2", filters=128, kernel_size=5, strides=(1, 1), padding="same", activation="relu", kernel_initializer=init)(pool1)

        # third layer: Conv (3x3) + Pool (2x2) + Simple Batch Norm - Output size: 200 x 16 x 128
        conv3 = tf.keras.layers.Conv2D(name="cnn_conv3", filters=128, kernel_size=3, strides=(1, 1), padding="same", activation="relu", kernel_initializer=init)(conv2)
        batch_norm = tf.keras.layers.BatchNormalization(name="cnn_batch_normalization1", axis=-1, epsilon=0.001, center=False, scale=False)(conv3)
        pool2 = tf.keras.layers.MaxPooling2D(name="cnn_max_pooling2", pool_size=(2, 2), strides=(2, 2), padding='valid')(batch_norm)

        # fourth layer: Conv (3x3) - Output size: 200 x 16 x 256
        conv4 = tf.keras.layers.Conv2D(name="cnn_conv4", filters=256, kernel_size=3, strides=(1, 1), padding="same", activation="relu", kernel_initializer=init)(pool2)

        # fifth layer: Conv (3x3) - Output size: 200 x 16 x 256
        conv5 = tf.keras.layers.Conv2D(name="cnn_conv5", filters=256, kernel_size=3, strides=(1, 1), padding="same", activation="relu", kernel_initializer=init)(conv4)

        # sixth layer: Conv (3x3) + Simple Batch Norm - Output size: 200 x 16 x 512
        conv6 = tf.keras.layers.Conv2D(name="cnn_conv6", filters=512, kernel_size=3, strides=(1, 1), padding="same", activation="relu", kernel_initializer=init)(conv5)
        batch_norm = tf.keras.layers.BatchNormalization(name="cnn_batch_normalization2", axis=-1, epsilon=0.001, center=False, scale=False)(conv6)
        pool3 = tf.keras.layers.MaxPooling2D(name="cnn_max_pooling3", pool_size=(1, 2), strides=(1, 2), padding='valid')(batch_norm)

        # seventh layer: Conv (3x3) + Pool (2x2) - Output size: 100 x 8 x 512
        conv7 = tf.keras.layers.Conv2D(name="cnn_conv7", filters=512, kernel_size=3, strides=(1, 1), padding="same", activation="relu", kernel_initializer=init)(pool3)
        self.cnn_out = tf.keras.layers.MaxPooling2D(name="cnn_max_pooling4", pool_size=(1, 2), strides=(1, 2), padding='valid')(conv7)

    def __setup_rnn(self, inputs):
        """RNN model"""

        # rnn layer: BiLSTM
        sliced = tf.slice(name="rnn_slice", input_=inputs, begin=[0, 0, 0, 0], size=[1, 100, 1, 512])
        squeezed = tf.squeeze(name="rnn_squeeze1", input=sliced, axis=[2])

        lstm = tf.keras.layers.LSTM(units=512, return_sequences=True, kernel_initializer='he_normal')
        bilstm = tf.keras.layers.Bidirectional(name="rnn_bilstm", layer=lstm)(squeezed)
        bilstm = tf.expand_dims(name="rnn_expand", input=bilstm, axis=2)

        # atrous dilation
        init = tf.random_normal_initializer(stddev=0.1)
        atrous = tf.keras.layers.Conv2D(name="rnn_atrous_conv", filters=22, kernel_size=1, padding="same", dilation_rate=1, activation='softmax', kernel_initializer=init)(bilstm)
        self.rnn_out = tf.squeeze(name="rnn_squeeze2", input=atrous, axis=[2])

    def __setup_ctc(self, args):
        """CTC model"""

        # calculate loss function
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

        self.ctc_out = tf.keras.layers.Lambda(name='ctc', function=ctc_lambda_func, output_shape=(1,))(args)
