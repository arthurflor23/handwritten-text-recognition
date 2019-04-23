"""Create the HTR deep learning model using tf.keras"""

import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def setup_cnn(inputs):
    """CNN model to HTR"""

    # first layer: Conv (5x5) + Pool (2x2) - Output size: 400 x 32 x 64
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal")(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)

    # second layer: Conv (5x5) - Output size: 400 x 32 x 128
    conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal")(pool1)

    # third layer: Conv (3x3) + Pool (2x2) + Simple Batch Norm - Output size: 200 x 16 x 128
    conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal")(conv2)
    batch_norm = tf.keras.layers.BatchNormalization(axis=-1, epsilon=0.001, center=False, scale=False)(conv3)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(batch_norm)

    # fourth layer: Conv (3x3) - Output size: 200 x 16 x 256
    conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal")(pool2)

    # fifth layer: Conv (3x3) - Output size: 200 x 16 x 256
    conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal")(conv4)

    # sixth layer: Conv (3x3) + Simple Batch Norm - Output size: 200 x 16 x 512
    conv6 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal")(conv5)
    batch_norm = tf.keras.layers.BatchNormalization(axis=-1, epsilon=0.001, center=False, scale=False)(conv6)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(batch_norm)

    # seventh layer: Conv (3x3) + Pool (2x2) - Output size: 100 x 8 x 512
    conv7 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal")(pool3)
    cnn_out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv7)

    return cnn_out


def setup_rnn(inputs):
    """RNN model to HTR"""

    rnnIn3d = tf.squeeze(tf.slice(inputs, [0, 0, 0, 0], [0, 100, 1, 512]))

    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))(rnnIn3d)
    concat = tf.expand_dims(bilstm, 2)

    rnn_out = tf.squeeze(tf.keras.layers.Conv2D(filters=22, kernel_size=1, dilation_rate=1, padding="same")(concat))
    
    return rnn_out


def setup_ctc(inputs):
    """CTC model to HTR"""

    # ctc_out = tf.keras.layers.Softmax(22, activation='softmax', kernel_initializer='he_normal')(inputs)

    ctcIn3d = tf.transpose(inputs, [1, 0, 2])

    softmax = tf.keras.layers.Softmax(axis=2)(ctcIn3d)
    # ctcOut = tf.keras.backend.ctc_decode(ctcIn3d, input_length=tf.size(softmax))
    
    # decoder = tf.keras.backend.ctc_decode()
    # decoder = tf.nn.ctc_greedy_decoder(inputs=ctcIn3d, sequence_length=tf.keras.backend.placeholder(tf.int32, [None]))

    # print("\n\Decoder:\n", ctc_decoder, "\n\n")
    return softmax


def ctc_decode(inputs):
    return tf.keras.backend.ctc_decode(y_pred=inputs, input_length=tf.size(inputs))


def ctc_loss(y_true, y_pred):
    """Create the loss function with CTC"""

    # y_pred = y_pred[:, 2:, :]
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, tf.size(y_true), tf.size(y_pred))
    return tf.reduce_mean(loss)


def HTR(input_shape):
    """HTR setup structure: CNN, RNN and CTC"""

    inputs = tf.keras.Input(shape=input_shape)
    cnn_out = setup_cnn(inputs)
    rnn_out = setup_rnn(cnn_out)
    ctc_out = setup_ctc(rnn_out)

    model = tf.keras.Model(name="HTR", inputs=inputs, outputs=ctc_out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=ctc_loss, metrics=["accuracy"])

    return model
