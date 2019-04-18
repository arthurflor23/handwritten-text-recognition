import tensorflow as tf

# First Layer: Conv (5x5) + Pool (2x2) - Output size: 400 x 32 x 64
cnnIn3d = tf.keras.backend.placeholder(dtype=tf.float32, shape=(50, 800, 64))
cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)

kernel = tf.Variable(tf.random.truncated_normal([5, 5, 1, 64], stddev=0.1))
conv = tf.nn.conv2d(input=cnnIn4d, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
# conv_norm = tf.compat.v1.layers.batch_normalization(conv)
relu = tf.nn.relu(conv)
pool = tf.nn.max_pool2d(input=relu, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')

print(pool)
### Tensor("MaxPool2d:0", shape=(50, 400, 32, 64), dtype=float32)


# strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
# kernel = tf.Variable(tf.random.truncated_normal([5, 5, 1, 64], stddev=0.1))
# conv = tf.nn.conv2d(input=cnnIn4d, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
# relu = tf.nn.relu(conv_norm)
# pool = tf.nn.max_pool2d(input=relu, ksize=(1, poolVals[0][0], poolVals[0][1], 1), strides=(1, strideVals[0][0], strideVals[0][1], 1), padding='VALID')

# print(pool)
### Tensor("MaxPool2d:0", shape=(50, 400, 32, 32), dtype=float32)


# inputs = tf.keras.layers.Input((800, 64, 1))
# conv1 = tf.keras.layers.Conv2D(64, 5, activation="relu", padding="same", strides=(1, 1))(inputs)
# conv1 = tf.keras.layers.Conv2D(64, 5, activation="relu", padding="same", strides=(1, 1))(conv1)
# pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(conv1)

# print(pool)
### Tensor("max_pooling2d/MaxPool:0", shape=(None, 400, 32, 64), dtype=float32)