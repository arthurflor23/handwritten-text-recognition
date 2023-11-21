import tensorflow as tf

from layers import SpectralSelfAttention
from layers import ConditionalBatchNormalization
from layers import SpectralNormalization
from layers import DynamicReshape


class SynthesisModel(tf.keras.Model):

    def __init__(self,
                 image_shape,
                 text_shape,
                 latent_dim,
                 vocab_dim,
                 embedding_dim,
                 channels,
                 g_blocks,
                 d_blocks,
                 **kwargs):

        super().__init__(**kwargs)

        self.generator = GeneratorModel(image_shape=image_shape,
                                        text_shape=text_shape,
                                        latent_dim=latent_dim,
                                        vocab_dim=vocab_dim,
                                        embedding_dim=embedding_dim,
                                        channels=channels,
                                        blocks=g_blocks)
        # self.generator.summary()

        self.discriminator = DiscriminatorModel(image_shape=image_shape,
                                                text_shape=text_shape,
                                                vocab_dim=vocab_dim,
                                                embedding_dim=embedding_dim,
                                                channels=channels,
                                                blocks=d_blocks)
        self.discriminator.summary()

        # self.patch_discriminator = None
        # self.style_encoder = None
        # self.style_backbone = None
        # self.writer_identifier = None
        # self.recognizer = None

    def compile(self, learning_rate=None):
        super().compile(run_eagerly=False)

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

        # self.generator.compile(loss='mse', optimizer=self.optimizer)

    def train_step(self, data):
        (image_inputs, texv_inputs), _ = data

    #     with tf.GradientTape() as tape:
    #         # Generate images
    #         generated_images = self.generator(image_inputs, texv_inputs)

    #         # Get the dynamic shape of the generated images
    #         dynamic_shape = tf.shape(generated_images)

    #         # Create dummy target data for loss calculation with the dynamic shape
    #         target = tf.random.normal(dynamic_shape, dtype=generated_images.dtype)

    #         # Simple loss function (mean squared error)
    #         loss = tf.reduce_mean(tf.square(generated_images - target))

    #     # Calculate gradients and update model weights
    #     gradients = tape.gradient(loss, self.generator.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

    #     return {"loss": loss}
        return {"loss": 0.0}


class GeneratorModel(tf.keras.Model):
    """
    A generator model that combines latent and vocabulary data for generative tasks.
    """

    def __init__(self,
                 image_shape,
                 text_shape,
                 latent_dim,
                 vocab_dim,
                 embedding_dim,
                 channels,
                 blocks,
                 **kwargs):
        """
        Initializes the generator model.

        Args:
            image_shape: list or tuple
                Shape of the output image.
            text_shape: list or tuple
                Shape of the input text.
            latent_dim: int
                Dimension of the latent space.
            vocab_dim: int
                Size of the vocabulary used in embeddings.
            embedding_dim: int
                Dimension of the embedding space.
            channels: int
                Base channels.
            blocks: list or tuple
                Blocks of channels.
            **kwargs
                Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.text_shape = text_shape
        self.latent_dim = latent_dim
        self.vocab_dim = vocab_dim
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.blocks = blocks

        self.build_model()

    def summary(self, line_length=None, positions=None, print_fn=None):
        """
        Prints a string summary of the network.

        Args:
            line_length: int, optional
                Total length of printed lines.
            positions: list of float, optional
                Positions of log elements in each line.
            print_fn: callable, optional
                Function used for printing the summary.
        """

        self.model.summary(line_length, positions, print_fn)

    def call(self, inputs, training=None, mask=None):
        """
        Executes the model on new inputs.

        Args:
            inputs: tensor or collection of tensors
                The input data to the model.
            training: bool, optional
                If True, the model is run in training mode.
            mask: tensor or collection of tensors, optional
                An optional mask (or masks) to be applied on the inputs.

        Returns:
            tensor or list of tensors
                The output from the model after processing the inputs.
        """

        return self.model(inputs, training, mask)

    def build_model(self):
        """
        Initializes the neural network model by defining its architecture.
        Sets `self.model` with the specified layers and configurations.
        """

        def residual_block_up(input_image, dense_latent, embedding, index, channels, blocks, upsample):

            chunk = tf.keras.layers.Lambda(
                lambda x: tf.split(x, num_or_size_splits=dense_latent.shape[-1]//blocks, axis=-1)[index],
                name=f'split_{index+1}')(dense_latent)

            chunk_concat = tf.keras.layers.Concatenate(axis=-1)([chunk, embedding])

            block_a = tf.keras.layers.UpSampling2D(size=upsample, interpolation='nearest')(input_image)
            block_a = SpectralNormalization(
                tf.keras.layers.Conv2D(channels, 1))(block_a)

            block_b = ConditionalBatchNormalization()([input_image, chunk_concat])
            block_b = tf.keras.layers.ReLU()(block_b)
            block_b = tf.keras.layers.UpSampling2D(size=upsample, interpolation='nearest')(block_b)
            block_b = SpectralNormalization(
                tf.keras.layers.Conv2D(channels, 3, padding='same'))(block_b)

            block_b = ConditionalBatchNormalization()([block_b, chunk_concat])
            block_b = tf.keras.layers.ReLU()(block_b)
            block_b = SpectralNormalization(
                tf.keras.layers.Conv2D(channels, 3, padding='same'))(block_b)

            block = tf.keras.layers.Add()([block_a, block_b])

            return block

        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,))
        latent_inputs = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=1), name='expand_dims')(latent_inputs)

        text_inputs = tf.keras.layers.Input(shape=self.text_shape)
        text_inputs = tf.keras.layers.Flatten()(text_inputs)

        text_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_dim + 1,
                                                   output_dim=self.embedding_dim,
                                                   mask_zero=True)(text_inputs)

        latent_tiled = tf.keras.layers.Lambda(
            lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], 1]), name='tile')([latent_inputs, text_embedding])

        latent_text_concat = tf.keras.layers.Concatenate(axis=-1)([latent_tiled, text_embedding])

        latent_dense = SpectralNormalization(tf.keras.layers.Dense(units=4*4*self.channels))(latent_text_concat)

        latent_feature_dense = SpectralNormalization(
            tf.keras.layers.Dense(units=self.latent_dim*len(self.blocks)))(latent_dense)

        latent_reshaped = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, [-1, tf.shape(x)[1]*4, 4, self.channels]), name='reshape')(latent_dense)

        block = tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 3, 2, 1]), name='transpose')(latent_reshaped)

        for i, x in enumerate(self.blocks):
            if i > 0 and i % 2 == 0:
                block = SpectralSelfAttention()(block)

            block = residual_block_up(input_image=block,
                                      dense_latent=latent_feature_dense,
                                      embedding=text_embedding,
                                      index=i,
                                      channels=x,
                                      blocks=len(self.blocks),
                                      upsample=(2, 2))

        outputs = tf.keras.layers.BatchNormalization()(block)
        outputs = tf.keras.layers.ReLU()(outputs)

        outputs = DynamicReshape(target_shape=self.image_shape)(outputs)

        outputs = SpectralNormalization(
            tf.keras.layers.Conv2D(1, 3, padding='same', activation='tanh'))(outputs)

        self.model = tf.keras.Model(inputs=[latent_inputs, text_inputs],
                                    outputs=outputs,
                                    name='generator')

    def get_config(self):
        """
        Returns the config of the model.

        Returns:
            A dictionary containing the configuration of the model.
        """

        config = {
            "image_shape": self.image_shape,
            "text_shape": self.text_shape,
            "latent_dim": self.latent_dim,
            "vocab_dim": self.vocab_dim,
            "embedding_dim": self.embedding_dim,
            "channels": self.channels,
            "blocks": self.blocks,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class DiscriminatorModel(tf.keras.Model):
    """
    A discriminator model that evaluates the authenticity of generated images.
    """

    def __init__(self,
                 image_shape,
                 text_shape,
                 vocab_dim,
                 embedding_dim,
                 channels,
                 blocks,
                 **kwargs):
        """
        Initializes the discriminator model.

        Args:
            image_shape: list or tuple
                Shape of the input image.
            text_shape: list or tuple
                Shape of the input text.
            vocab_dim: int
                Size of the vocabulary used in embeddings.
            embedding_dim: int
                Dimension of the embedding space.
            channels: int
                Base channels.
            blocks: list or tuple
                Blocks of channels.
            **kwargs
                Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.text_shape = text_shape
        self.vocab_dim = vocab_dim
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.blocks = blocks

        self.build_model()

    def summary(self, line_length=None, positions=None, print_fn=None):
        """
        Prints a string summary of the network.

        Args:
            line_length: int, optional
                Total length of printed lines.
            positions: list of float, optional
                Positions of log elements in each line.
            print_fn: callable, optional
                Function used for printing the summary.
        """

        self.model.summary(line_length, positions, print_fn)

    def call(self, inputs, training=None, mask=None):
        """
        Executes the model on new inputs.

        Args:
            inputs: tensor or collection of tensors
                The input data to the model.
            training: bool, optional
                If True, the model is run in training mode.
            mask: tensor or collection of tensors, optional
                An optional mask (or masks) to be applied on the inputs.

        Returns:
            tensor or list of tensors
                The output from the model after processing the inputs.
        """

        return self.model(inputs, training, mask)

    def build_model(self):
        """
        Initializes the neural network model by defining its architecture.
        Sets `self.model` with the specified layers and configurations.
        """

        def residual_block_down(input_image, channels, downsample):

            block_a = SpectralNormalization(
                tf.keras.layers.Conv2D(channels, 1))(input_image)

            if downsample:
                block_a = tf.keras.layers.AveragePooling2D()(block_a)

            block_b = tf.keras.layers.ReLU()(input_image)
            block_b = SpectralNormalization(
                tf.keras.layers.Conv2D(channels, 3, padding='same'))(block_b)

            block_b = tf.keras.layers.ReLU()(block_b)
            block_b = SpectralNormalization(
                tf.keras.layers.Conv2D(channels, 3, padding='same'))(block_b)

            if downsample:
                block_b = tf.keras.layers.AveragePooling2D()(block_b)

            block = tf.keras.layers.Add()([block_a, block_b])

            return block

        image_inputs = tf.keras.layers.Input(shape=self.image_shape)

        text_inputs = tf.keras.layers.Input(shape=self.text_shape)
        text_inputs = tf.keras.layers.Flatten()(text_inputs)

        text_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_dim + 1,
                                                   output_dim=self.embedding_dim,
                                                   mask_zero=True)(text_inputs)
        text_embedding = tf.keras.layers.Flatten()(text_embedding)

        block = SpectralNormalization(
            tf.keras.layers.Conv2D(self.channels, 3, 1, padding='same'))(image_inputs)

        for i, x in enumerate(self.blocks):
            if i % 2 == 0:
                block = SpectralSelfAttention()(block)

            block = residual_block_down(input_image=block,
                                        channels=x,
                                        downsample=i < len(self.blocks) - 1)

        outputs = tf.keras.layers.GlobalAveragePooling2D()(block)
        outputs = tf.keras.layers.Concatenate(axis=-1)([text_embedding, outputs])

        outputs = SpectralNormalization(
            tf.keras.layers.Dense(units=1))(outputs)

        self.model = tf.keras.Model(inputs=[image_inputs, text_inputs],
                                    outputs=outputs,
                                    name='discriminator')

    def get_config(self):
        """
        Returns the config of the model.

        Returns:
            A dictionary containing the configuration of the model.
        """

        config = {
            "image_shape": self.image_shape,
            "text_shape": self.text_shape,
            "vocab_dim": self.vocab_dim,
            "embedding_dim": self.embedding_dim,
            "channels": self.channels,
            "blocks": self.blocks,
        }
        base_config = super().get_config()
        return {**base_config, **config}
