import tensorflow as tf

from layers import SpectralSelfAttention, ConditionalBatchNormalization


class SynthesisModel(tf.keras.Model):

    def __init__(self,
                 vocab_size=80,
                 embedding_dim=120,
                 style_dim=32):

        super(SynthesisModel, self).__init__()

        self.generator = GeneratorModule(vocab_size, embedding_dim, style_dim)
        self.discriminator = None
        self.patch_discriminator = None
        self.style_encoder = None
        self.style_backbone = None
        self.writer_identifier = None
        self.recognizer = None

    def compile(self, learning_rate=None):
        super(SynthesisModel, self).compile()

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

    def train_step(self, data):
        (image_inputs, text_inputs), _ = data

        with tf.GradientTape() as tape:
            # Generate images
            generated_images = self.generator(image_inputs, text_inputs)

            # Get the dynamic shape of the generated images
            dynamic_shape = tf.shape(generated_images)

            # Create dummy target data for loss calculation with the dynamic shape
            target = tf.random.normal(dynamic_shape, dtype=generated_images.dtype)

            # Simple loss function (mean squared error)
            loss = tf.reduce_mean(tf.square(generated_images - target))

        # Calculate gradients and update model weights
        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return {"loss": loss}


class GeneratorModule(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 style_dim):

        super(GeneratorModule, self).__init__()

        self.text_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size + 1, output_dim=embedding_dim, mask_zero=True)

        # self.filter_sn_dense = tf.keras.layers.SpectralNormalization(
        #     tf.keras.layers.Dense(units=embedding_dim + style_dim), power_iterations=512 * 16)

        # self.style_sn_dense = tf.keras.layers.SpectralNormalization(
        #     tf.keras.layers.Dense(units=style_dim), power_iterations=style_dim * 4)

        # # blocks
        # self.block1 = GeneratorBlock(units=256, upsample=(2, 1))
        # self.block2 = GeneratorBlock(units=128, upsample=(2, 2))
        # self.block3 = GeneratorBlock(units=64, upsample=(2, 2))
        # self.block4 = GeneratorBlock(units=64, upsample=(2, 2))

        # self.attention = SpectralSelfAttention(units=64)

        # self.outlayer = tf.keras.Sequential([
        #     tf.keras.layers.BatchNormalization(),
        #     # ConditionalBatchNormalization(units=64),
        #     tf.keras.layers.ReLU(),
        #     tf.keras.layers.SpectralNormalization(
        #         tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same'))
        # ])

        self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same')

    def call(self, image_inputs, text_inputs, training=True):

        # # Process text inputs through the embedding layer
        # text_embedding = self.text_embedding(text_inputs)

        # # Filter processing
        # filter_processed = self.filter_sn_dense(text_embedding)

        # # Style processing (if needed, depending on how style is derived from text)
        # style_processed = self.style_sn_dense(text_embedding)

        # # Combine the filter and style processed outputs with the image inputs
        # x = tf.concat([image_inputs, filter_processed, style_processed], axis=-1)

        # # Passing through Generator Blocks
        # x = self.block1(x, training=training)
        # x = self.block2(x, training=training)
        # x = self.block3(x, training=training)
        # x = self.block4(x, training=training)

        # # Applying Attention
        # x = self.attention(x)

        # # Final Output Layer
        # x = self.outlayer(x, training=training)

        x = self.conv(image_inputs)

        return x


class GeneratorBlock(tf.keras.layers.Layer):

    def __init__(self, units, upsample=None):
        super(GeneratorBlock, self).__init__()

        self.units = units
        self.upsample = upsample

        self.conv1 = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=units, kernel_size=3, padding='same'))

        self.conv2 = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=units, kernel_size=3, padding='same'))

        self.conv3 = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=units, kernel_size=1, padding='same'))

        self.bn1 = ConditionalBatchNormalization(units=units)
        self.bn2 = ConditionalBatchNormalization(units=units)

        self.activation = tf.keras.layers.ReLU()

    def call(self, x, y, training=True):
        h = self.activation(self.bn1(x, y, training=training))

        if self.upsample is not None:
            new_height = int(x.shape[1] * self.upsample[0])
            new_width = int(x.shape[2] * self.upsample[1])
            h = tf.image.resize(h, [new_height, new_width])
            x = tf.image.resize(x, [new_height, new_width])

        h = self.conv1(h)
        h = self.activation(self.bn2(h, y, training=training))
        h = self.conv2(h)

        x = self.conv3(x)

        return h + x
