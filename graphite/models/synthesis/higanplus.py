import tensorflow as tf

from layers import SpectralSelfAttention
from layers import ConditionalBatchNormalization
from layers import SpectralNormalization
from layers import DynamicReshape


class SynthesisModel(tf.keras.Model):

    def __init__(self,
                 image_shape,
                 latent_dim,
                 vocab_dim,
                 embedding_dim,
                 dense_dim,
                 blocks,
                 **kwargs):

        super().__init__(**kwargs)

        self.generator = GeneratorModel(image_shape=image_shape,
                                        latent_dim=latent_dim,
                                        vocab_dim=vocab_dim,
                                        embedding_dim=embedding_dim,
                                        dense_dim=dense_dim,
                                        blocks=blocks)
        self.generator.summary()

        self.discriminator = DiscriminatorModel()
        # self.discriminator.summary()

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
                 vocab_dim,
                 latent_dim,
                 embedding_dim,
                 dense_dim,
                 blocks,
                 **kwargs):
        """
        Initializes the generator model.

        Args:
            image_shape: list or tuple
                Shape of the output image.
            vocab_dim: int
                Size of the vocabulary used in embeddings.
            latent_dim: int
                Dimension of the latent space.
            embedding_dim: int
                Dimension of the embedding space.
            dense_dim: int
                Dimension for the dense space.
            blocks: list or tuple
                Blocks of channels.
            **kwargs
                Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.vocab_dim = vocab_dim
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.dense_dim = dense_dim
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
        Builds the generator model with a series of layers including residual blocks,
        batch normalization, and convolutional layers.

        The model processes input data through these layers to generate an output image.
        """

        def residual_block_up(dense, dense_latent, embedding, index, channels, blocks, upsampling=(1, 1)):
            """
            Constructs an upsampling residual block within the generator model.

            Args:
                dense: tensor
                    Input tensor for the residual block.
                dense_latent: tensor
                    Latent feature tensor to be split and concatenated in the block.
                embedding: tensor
                    Embedding tensor to be concatenated with latent features.
                index: int
                    Index for splitting the latent features.
                channels: int
                    Number of output channels for the convolutions.
                blocks: int
                    Total number of blocks to determine the split size.
                upsampling: tuple, optional
                    Upsampling factor for spatial dimensions.

            Returns:
                tensor
                    Output tensor of the residual block.
            """

            chunk = tf.keras.layers.Lambda(
                lambda x: tf.split(x, num_or_size_splits=dense_latent.shape[-1]//blocks, axis=-1)[index],
                name=f'split_{index+1}')(dense_latent)

            dense_concat = tf.keras.layers.Concatenate(axis=-1)([chunk, embedding])

            block_a = tf.keras.layers.UpSampling2D(size=upsampling, interpolation='nearest')(dense)
            block_a = SpectralNormalization(
                tf.keras.layers.Conv2D(channels, 1))(block_a)

            block_b = ConditionalBatchNormalization()([dense, dense_concat])
            block_b = tf.keras.layers.ReLU()(block_b)
            block_b = tf.keras.layers.UpSampling2D(size=upsampling, interpolation='nearest')(block_b)
            block_b = SpectralNormalization(
                tf.keras.layers.Conv2D(channels, 3, padding='same'))(block_b)

            block_b = ConditionalBatchNormalization()([block_b, dense_concat])
            block_b = tf.keras.layers.ReLU()(block_b)
            block_b = SpectralNormalization(
                tf.keras.layers.Conv2D(channels, 3, padding='same'))(block_b)

            block = tf.keras.layers.Add()([block_a, block_b])

            return block

        vocab_inputs = tf.keras.layers.Input(shape=(self.vocab_dim,))
        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,))

        vocab_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_dim + 1,
                                                    output_dim=self.embedding_dim,
                                                    mask_zero=True)(vocab_inputs)

        latent_expanded = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=1), name='expand_dims')(latent_inputs)

        latent_tiled = tf.keras.layers.Lambda(
            lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], 1]), name='tile')([latent_expanded, vocab_embedding])

        latent_vocab_concat = tf.keras.layers.Concatenate(axis=-1)([latent_tiled, vocab_embedding])

        latent_dense = SpectralNormalization(tf.keras.layers.Dense(units=4*4*self.dense_dim))(latent_vocab_concat)

        latent_feature_dense = SpectralNormalization(
            tf.keras.layers.Dense(units=self.latent_dim*len(self.blocks)))(latent_dense)

        latent_reshaped = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, [-1, tf.shape(x)[1]*4, 4, self.dense_dim]), name='reshape')(latent_dense)

        block = tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 3, 2, 1]), name='transpose')(latent_reshaped)

        for i, x in enumerate(self.blocks):
            if (i > 0) and (i % 2 == 0):
                block = SpectralSelfAttention()(block)

            block = residual_block_up(dense=block,
                                      dense_latent=latent_feature_dense,
                                      embedding=vocab_embedding,
                                      index=i,
                                      channels=x,
                                      blocks=len(self.blocks),
                                      upsampling=(2, 2))

        outputs = tf.keras.layers.BatchNormalization()(block)
        outputs = tf.keras.layers.ReLU()(outputs)

        outputs = DynamicReshape(target_shape=self.image_shape)(outputs)

        outputs = SpectralNormalization(
            tf.keras.layers.Conv2D(1, 3, padding='same', activation='tanh'))(outputs)

        self.model = tf.keras.Model(inputs=[latent_inputs, vocab_inputs], outputs=outputs)

    def get_config(self):
        """
        Returns the config of the model.

        Returns:
            A dictionary containing the configuration of the model.
        """

        config = {
            "image_shape": self.image_shape,
            "vocab_dim": self.vocab_dim,
            "latent_dim": self.latent_dim,
            "embedding_dim": self.embedding_dim,
            "dense_dim": self.dense_dim,
            "blocks": self.blocks,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class DiscriminatorModel(tf.keras.Model):
    """
    A discriminator model that evaluates the authenticity of generated images.
    """

    def __init__(self, **kwargs):
        """
        Initializes the discriminator model.

        Args:
            **kwargs
                Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

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
        Builds the generator model with a series of layers including residual blocks,
        batch normalization, and convolutional layers.

        The model processes input data through these layers to generate an output image.
        """

        pass

    def get_config(self):
        """
        Returns the config of the model.

        Returns:
            A dictionary containing the configuration of the model.
        """

        config = {
            # "blocks": self.blocks,
        }
        base_config = super().get_config()
        return {**base_config, **config}
