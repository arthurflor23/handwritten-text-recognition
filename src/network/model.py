"""Create the HTR deep learning model using tf.keras"""

import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# inputs = tf.keras.backend.placeholder(dtype=tf.float32, shape=(50, 800, 64))
# inputs = tf.expand_dims(input=inputs, axis=3)
# # print(inputs)

# kernel = tf.Variable(tf.random.truncated_normal([5, 5, 1, 64], stddev=0.1))
# conv = tf.nn.conv2d(input=inputs, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
# print(conv)

# relu = tf.nn.relu(conv)
# print(relu)

# pool = tf.nn.max_pool2d(input=relu, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')
# print(pool)
### Tensor("ExpandDims:0", shape=(50, 800, 64, 1), dtype=float32)
### Tensor("Conv2D:0", shape=(50, 800, 64, 64), dtype=float32)
### Tensor("Relu:0", shape=(50, 800, 64, 64), dtype=float32)
### Tensor("MaxPool2d:0", shape=(50, 400, 32, 64), dtype=float32)


def HTR(input_shape):
    """HTR structure: CNN, RNN and CTC"""

    ### Begin CNN ###
    # first layer: Conv (5x5) + Pool (2x2) - Output size: 400 x 32 x 64
    cnnIn3d = tf.keras.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal")(cnnIn3d)
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
    cnnOut3d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv7)


    ### Begin RNN ###
    rnnIn3d = tf.squeeze(tf.slice(cnnOut3d, [0, 0, 0, 0], [0, 100, 1, 512]))

    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))(rnnIn3d)
    concat = tf.expand_dims(bilstm, 2)

    rnnOut3d = tf.squeeze(tf.keras.layers.Conv2D(filters=22, kernel_size=1, dilation_rate=1, padding="same")(concat))


    ### Begin CTC ###
    ctcIn3d = tf.transpose(rnnOut3d, [1, 0, 2])

    # softmax = tf.keras.layers.Softmax(axis=2)(ctcIn3d)
    # ctc_decoder = tf.keras.backend.ctc_decode(softmax, tf.size(softmax))
    
    # print("\n\Decoder:\n", ctc_decoder, "\n\n")


    model = tf.keras.Model(name="HTR", inputs=cnnIn3d, outputs=ctcIn3d)
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=ctc_loss, metrics=["accuracy"])

    return model


def ctc_loss(y_true, y_pred):
    # y_pred = y_pred[:, 2:, :]
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, tf.size(y_true), tf.size(y_pred))
    return tf.reduce_mean(loss)
