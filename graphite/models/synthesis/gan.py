import tensorflow as tf

from models.components import SpectralSelfAttention
from models.components import ConditionalBatchNormalization
from models.components import SpectralNormalization
from models.components import ExtractPatches
from models.components import CTCLoss
from models.components import CXLoss


class SynthesisModel(tf.keras.Model):

    def __init__(self,
                 image_shape,
                 patch_shape,
                 lexical_shape,
                 latent_dim,
                 writer_dim,
                 embedding_dim,
                 backbone_dim,
                 generator_blocks,
                 discriminator_blocks,
                 **kwargs):

        super().__init__(**kwargs)

        self.generator = GeneratorModel(image_shape=image_shape,
                                        lexical_shape=lexical_shape,
                                        latent_dim=latent_dim,
                                        embedding_dim=embedding_dim,
                                        blocks=generator_blocks,
                                        name='generator')
        # self.generator.summary()

        self.discriminator = DiscriminatorModel(image_shape=image_shape,
                                                patch_shape=None,
                                                lexical_shape=lexical_shape,
                                                embedding_dim=embedding_dim,
                                                blocks=discriminator_blocks,
                                                name='discriminator')
        # self.discriminator.summary()

        self.patch_discriminator = DiscriminatorModel(image_shape=image_shape,
                                                      patch_shape=patch_shape,
                                                      lexical_shape=lexical_shape,
                                                      embedding_dim=embedding_dim,
                                                      blocks=discriminator_blocks,
                                                      name='patch_discriminator')
        # self.patch_discriminator.summary()

        self.style_backbone = StyleBackboneModel(image_shape=image_shape,
                                                 filters=backbone_dim,
                                                 name='style_backbone')
        # self.style_backbone.summary()

        self.style_encoder = StyleEncoderModel(features_shape=self.style_backbone.features_shape,
                                               latent_dim=latent_dim,
                                               name='style_encoder')
        # self.style_encoder.summary()

        self.identifier = IdentifierModel(features_shape=self.style_backbone.features_shape,
                                          writer_dim=writer_dim,
                                          name='identifier')
        # self.identifier.summary()

        self.recognizer = RecognizerModel(features_shape=self.style_backbone.features_shape,
                                          lexical_shape=lexical_shape,
                                          name='recognizer')
        # self.recognizer.summary()

    def compile(self, learning_rate=0.001):

        super().compile(run_eagerly=False)

        self.generator_optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

        self.cx_loss = CXLoss()
        self.ctc_loss = CTCLoss()
        self.kld_loss = tf.keras.losses.KLDivergence()
        self.classify_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def train_step(self, data):
        (image_inputs, text_inputs, writer_inputs), _ = data

        with tf.GradientTape(persistent=True) as tape:
            # Forward pass through the generator to create fake images
            fake_images = self.generator([image_inputs, text_inputs], training=True)

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
                 lexical_shape,
                 latent_dim,
                 embedding_dim,
                 blocks,
                 **kwargs):
        """
        Initializes the model class.

        Args:
            image_shape: list or tuple
                Shape of the output image.
            lexical_shape: list or tuple
                Shape of the text sequences and vocabulary encoding.
            latent_dim: int
                Dimension of the latent space.
            embedding_dim: int
                Dimension of the embedding space.
            blocks: list or tuple
                Blocks of channels.
            **kwargs
                Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.blocks = blocks

        self.build_model()

    def get_config(self):
        """
        Returns the config of the model.

        Returns:
            A dictionary containing the configuration of the model.
        """

        config = {
            "image_shape": self.image_shape,
            "lexical_shape": self.lexical_shape,
            "latent_dim": self.latent_dim,
            "embedding_dim": self.embedding_dim,
            "blocks": self.blocks,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def summary(self,
                line_length=None,
                positions=None,
                print_fn=None,
                expand_nested=False,
                show_trainable=False,
                layer_range=None):
        """
        Prints a string summary of the network.

        Args:
            line_length: int, optional
                Total length of printed lines.
            positions: list of float, optional
                Positions of log elements in each line.
            print_fn: callable, optional
                Function used for printing the summary.
            expand_nested: bool, optional
                Whether to expand the nested models.
            show_trainable: bool, optional
                Whether to show if a layer is trainable.
            layer_range: list or tuple, optional
                Range of layers to include in the model summary.
        """

        self.model.summary(line_length,
                           positions,
                           print_fn,
                           expand_nested,
                           show_trainable,
                           layer_range)

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

        def residual_block_up(input_image, dense_latent, embedding, index, filters, blocks, upsample):

            chunk = tf.keras.layers.Lambda(
                lambda x: tf.split(x, num_or_size_splits=dense_latent.shape[-1]//blocks, axis=-1)[index],
                name=f'split_{index+1}')(dense_latent)

            chunk_concat = tf.keras.layers.Concatenate(axis=-1)([chunk, embedding])

            block_a = tf.keras.layers.UpSampling2D(size=upsample, interpolation='nearest')(input_image)
            block_a = SpectralNormalization(
                tf.keras.layers.Conv2D(filters, kernel_size=1))(block_a)

            block_b = ConditionalBatchNormalization()([input_image, chunk_concat])
            block_b = tf.keras.layers.ReLU()(block_b)
            block_b = tf.keras.layers.UpSampling2D(size=upsample, interpolation='nearest')(block_b)
            block_b = SpectralNormalization(
                tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(block_b)

            block_b = ConditionalBatchNormalization()([block_b, chunk_concat])
            block_b = tf.keras.layers.ReLU()(block_b)
            block_b = SpectralNormalization(
                tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(block_b)

            block = tf.keras.layers.Add()([block_a, block_b])

            return block

        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,))

        latent_inputs = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=1), name='expand_dims')(latent_inputs)

        text_inputs = tf.keras.layers.Input(shape=self.lexical_shape[:-1])
        text_inputs = tf.keras.layers.Flatten()(text_inputs)

        text_embedding = tf.keras.layers.Embedding(input_dim=self.lexical_shape[-1] + 1,
                                                   output_dim=self.embedding_dim,
                                                   mask_zero=True)(text_inputs)

        latent_tiled = tf.keras.layers.Lambda(
            lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], 1]), name='tile')([latent_inputs, text_embedding])

        latent_text_concat = tf.keras.layers.Concatenate(axis=-1)([latent_tiled, text_embedding])

        latent_dense = SpectralNormalization(tf.keras.layers.Dense(units=4*4*self.blocks[0]))(latent_text_concat)

        latent_feature_dense = SpectralNormalization(
            tf.keras.layers.Dense(units=self.latent_dim*len(self.blocks)))(latent_dense)

        latent_reshaped = tf.keras.layers.Reshape(target_shape=(latent_dense.get_shape()[1]*4, 4, -1))(latent_dense)

        block = tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 3, 2, 1]), name='transpose')(latent_reshaped)

        for i, x in enumerate(self.blocks):
            if i > 0 and i % 2 == 0:
                block = SpectralSelfAttention()(block)

            block = residual_block_up(input_image=block,
                                      dense_latent=latent_feature_dense,
                                      embedding=text_embedding,
                                      index=i,
                                      filters=x,
                                      blocks=len(self.blocks),
                                      upsample=(2, 2))

        outputs = tf.keras.layers.BatchNormalization()(block)
        outputs = tf.keras.layers.ReLU()(outputs)

        outputs = tf.keras.layers.Reshape(target_shape=(self.image_shape[0], self.image_shape[1], -1))(outputs)

        outputs = SpectralNormalization(
            tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', activation='tanh'))(outputs)

        self.model = tf.keras.Model(inputs=[latent_inputs, text_inputs], outputs=outputs, name=self.name)


class DiscriminatorModel(tf.keras.Model):
    """
    A discriminator model that evaluates the authenticity of generated images.
    """

    def __init__(self,
                 image_shape,
                 patch_shape,
                 lexical_shape,
                 embedding_dim,
                 blocks,
                 **kwargs):
        """
        Initializes the model class.

        Args:
            image_shape: list or tuple
                Shape of the input image.
            patch_shape: list, tuple or None
                Defines whether to apply patches.
            lexical_shape: list or tuple
                Shape of the text sequences and vocabulary encoding.
            embedding_dim: int
                Dimension of the embedding space.
            blocks: list or tuple
                Blocks of channels.
            blocks: list or tuple
                Blocks of channels.
            **kwargs
                Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.lexical_shape = lexical_shape
        self.embedding_dim = embedding_dim
        self.blocks = blocks

        self.build_model()

    def get_config(self):
        """
        Returns the config of the model.

        Returns:
            A dictionary containing the configuration of the model.
        """

        config = {
            "image_shape": self.image_shape,
            "patch_shape": self.patch_shape,
            "lexical_shape": self.lexical_shape,
            "embedding_dim": self.embedding_dim,
            "blocks": self.blocks,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def summary(self,
                line_length=None,
                positions=None,
                print_fn=None,
                expand_nested=False,
                show_trainable=False,
                layer_range=None):
        """
        Prints a string summary of the network.

        Args:
            line_length: int, optional
                Total length of printed lines.
            positions: list of float, optional
                Positions of log elements in each line.
            print_fn: callable, optional
                Function used for printing the summary.
            expand_nested: bool, optional
                Whether to expand the nested models.
            show_trainable: bool, optional
                Whether to show if a layer is trainable.
            layer_range: list or tuple, optional
                Range of layers to include in the model summary.
        """

        self.model.summary(line_length,
                           positions,
                           print_fn,
                           expand_nested,
                           show_trainable,
                           layer_range)

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

        def residual_block_down(input_image, filters, downsample):

            block_a = SpectralNormalization(
                tf.keras.layers.Conv2D(filters, kernel_size=1))(input_image)

            if downsample:
                block_a = tf.keras.layers.AveragePooling2D()(block_a)

            block_b = tf.keras.layers.ReLU()(input_image)
            block_b = SpectralNormalization(
                tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(block_b)

            block_b = tf.keras.layers.ReLU()(block_b)
            block_b = SpectralNormalization(
                tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(block_b)

            if downsample:
                block_b = tf.keras.layers.AveragePooling2D()(block_b)

            block = tf.keras.layers.Add()([block_a, block_b])

            return block

        image_inputs = tf.keras.layers.Input(shape=self.image_shape)

        text_inputs = tf.keras.layers.Input(shape=self.lexical_shape[:-1])
        text_inputs = tf.keras.layers.Flatten()(text_inputs)

        text_embedding = tf.keras.layers.Embedding(input_dim=self.lexical_shape[-1] + 1,
                                                   output_dim=self.embedding_dim,
                                                   mask_zero=True)(text_inputs)
        text_embedding = tf.keras.layers.Flatten()(text_embedding)

        if self.patch_shape is None:
            block = SpectralNormalization(
                tf.keras.layers.Conv2D(self.blocks[-1], kernel_size=3, strides=1, padding='same'))(image_inputs)
        else:
            patch_inputs = ExtractPatches(patch_shape=self.patch_shape)(image_inputs)

            block = SpectralNormalization(
                tf.keras.layers.Conv2D(self.blocks[-1], kernel_size=3, strides=1, padding='same'))(patch_inputs)

        for i, x in enumerate(self.blocks):
            if i % 2 == 0:
                block = SpectralSelfAttention()(block)

            block = residual_block_down(input_image=block,
                                        filters=x,
                                        downsample=i < len(self.blocks) - 1)

        outputs = tf.keras.layers.GlobalAveragePooling2D()(block)
        outputs = tf.keras.layers.Concatenate(axis=-1)([text_embedding, outputs])

        outputs = SpectralNormalization(
            tf.keras.layers.Dense(units=1))(outputs)

        self.model = tf.keras.Model(inputs=[image_inputs, text_inputs], outputs=outputs, name=self.name)


class StyleBackboneModel(tf.keras.Model):
    """
    A backbone model that extracts style patterns from images.
    """

    def __init__(self,
                 image_shape,
                 filters,
                 **kwargs):
        """
        Initializes the model class.

        Args:
            image_shape: list or tuple
                Shape of the input image.
            filters: int
                Number of filters used in the first convolutional layers.
            **kwargs
                Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.features_shape = None
        self.filters = filters

        self.build_model()

    def get_config(self):
        """
        Returns the config of the model.

        Returns:
            A dictionary containing the configuration of the model.
        """

        config = {
            "image_shape": self.image_shape,
            "features_shape": self.features_shape,
            "filters": self.filters,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def summary(self,
                line_length=None,
                positions=None,
                print_fn=None,
                expand_nested=False,
                show_trainable=False,
                layer_range=None):
        """
        Prints a string summary of the network.

        Args:
            line_length: int, optional
                Total length of printed lines.
            positions: list of float, optional
                Positions of log elements in each line.
            print_fn: callable, optional
                Function used for printing the summary.
            expand_nested: bool, optional
                Whether to expand the nested models.
            show_trainable: bool, optional
                Whether to show if a layer is trainable.
            layer_range: list or tuple, optional
                Range of layers to include in the model summary.
        """

        self.model.summary(line_length,
                           positions,
                           print_fn,
                           expand_nested,
                           show_trainable,
                           layer_range)

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

        image_inputs = tf.keras.layers.Input(shape=self.image_shape)

        feats = []
        filters = self.filters

        conv = tf.keras.layers.Conv2D(filters, kernel_size=5, strides=2, padding='same')(image_inputs)

        for _ in range(4):
            block1 = tf.keras.layers.ReLU()(conv)
            block1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(block1)
            block1 = tf.keras.layers.BatchNormalization()(block1)

            block1 = tf.keras.layers.ReLU()(block1)
            block1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(block1)
            block1 = tf.keras.layers.BatchNormalization()(block1)

            conv = tf.keras.layers.Add()([conv, block1])

            block2 = tf.keras.layers.ReLU()(conv)
            block2 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(block2)
            block2 = tf.keras.layers.BatchNormalization()(block2)

            block2 = tf.keras.layers.ReLU()(block2)
            block2 = tf.keras.layers.Conv2D(filters*2, kernel_size=3, strides=1, padding='same')(block2)
            block2 = tf.keras.layers.BatchNormalization()(block2)

            shortcut = tf.keras.layers.Conv2D(filters*2,
                                              kernel_size=1,
                                              strides=1,
                                              padding='valid',
                                              use_bias=False)(conv)

            conv = tf.keras.layers.Add()([shortcut, block2])
            conv = tf.keras.layers.ZeroPadding2D(padding=1)(conv)
            conv = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(conv)

            feats.append(conv)
            filters *= 2

        conv = tf.keras.layers.ReLU()(conv)
        conv = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.ReLU()(conv)

        outputs = tf.keras.layers.Reshape(target_shape=(conv.get_shape()[1]*conv.get_shape()[2], -1))(conv)

        self.features_shape = outputs.get_shape()[1:]
        self.model = tf.keras.Model(inputs=image_inputs, outputs=[outputs, feats], name=self.name)


class StyleEncoderModel(tf.keras.Model):
    """
    An encoder model that encodes extracted style features from images into a representative style vector.
    """

    def __init__(self,
                 features_shape,
                 latent_dim,
                 **kwargs):
        """
        Initializes the model class.

        Args:
            features_shape: list or tuple
                Shape of the input features.
            latent_dim: int
                Dimension of the latent space.
            **kwargs
                Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.features_shape = features_shape
        self.latent_dim = latent_dim

        self.build_model()

    def get_config(self):
        """
        Returns the config of the model.

        Returns:
            A dictionary containing the configuration of the model.
        """

        config = {
            "features_shape": self.features_shape,
            "latent_dim": self.latent_dim,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def summary(self,
                line_length=None,
                positions=None,
                print_fn=None,
                expand_nested=False,
                show_trainable=False,
                layer_range=None):
        """
        Prints a string summary of the network.

        Args:
            line_length: int, optional
                Total length of printed lines.
            positions: list of float, optional
                Positions of log elements in each line.
            print_fn: callable, optional
                Function used for printing the summary.
            expand_nested: bool, optional
                Whether to expand the nested models.
            show_trainable: bool, optional
                Whether to show if a layer is trainable.
            layer_range: list or tuple, optional
                Range of layers to include in the model summary.
        """

        self.model.summary(line_length,
                           positions,
                           print_fn,
                           expand_nested,
                           show_trainable,
                           layer_range)

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

        feature_inputs = tf.keras.layers.Input(shape=self.features_shape)

        style = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=-2) / tf.cast(tf.shape(x)[-2], tf.float32) + 1e-8,
            name='reduce')(feature_inputs)

        style_dense = tf.keras.layers.Dense(self.features_shape[-1])(style)
        style_dense = tf.keras.layers.LeakyReLU(alpha=0.01)(style_dense)

        style_dense = tf.keras.layers.Dense(self.features_shape[-1])(style_dense)
        style_dense = tf.keras.layers.LeakyReLU(alpha=0.01)(style_dense)

        mu = tf.keras.layers.Dense(self.latent_dim)(style_dense)
        logvar = tf.keras.layers.Dense(self.latent_dim)(style_dense)

        outputs = tf.keras.layers.Lambda(
            lambda x: x[0] + tf.exp(0.5 * x[1]) * tf.random.normal(tf.shape(x[1])),
            name='reparameterize')([mu, logvar])

        self.model = tf.keras.Model(inputs=feature_inputs, outputs=[outputs, mu, logvar], name=self.name)


class IdentifierModel(tf.keras.Model):
    """
    A writer identifier model that classifies the handwriting image based in the extracted style features.
    """

    def __init__(self,
                 features_shape,
                 writer_dim,
                 **kwargs):
        """
        Initializes the model class.

        Args:
            features_shape: list or tuple
                Shape of the input features.
            writer_dim: int
                Number of writers.
            **kwargs
                Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.features_shape = features_shape
        self.writer_dim = writer_dim

        self.build_model()

    def get_config(self):
        """
        Returns the config of the model.

        Returns:
            A dictionary containing the configuration of the model.
        """

        config = {
            "features_shape": self.features_shape,
            "writer_dim": self.writer_dim,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def summary(self,
                line_length=None,
                positions=None,
                print_fn=None,
                expand_nested=False,
                show_trainable=False,
                layer_range=None):
        """
        Prints a string summary of the network.

        Args:
            line_length: int, optional
                Total length of printed lines.
            positions: list of float, optional
                Positions of log elements in each line.
            print_fn: callable, optional
                Function used for printing the summary.
            expand_nested: bool, optional
                Whether to expand the nested models.
            show_trainable: bool, optional
                Whether to show if a layer is trainable.
            layer_range: list or tuple, optional
                Range of layers to include in the model summary.
        """

        self.model.summary(line_length,
                           positions,
                           print_fn,
                           expand_nested,
                           show_trainable,
                           layer_range)

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

        feature_inputs = tf.keras.layers.Input(shape=self.features_shape)

        style = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=-2) / tf.cast(tf.shape(x)[-2], tf.float32) + 1e-8,
            name='reduce')(feature_inputs)

        style_dense = tf.keras.layers.Dense(self.features_shape[-1])(style)
        style_dense = tf.keras.layers.LeakyReLU(alpha=0.01)(style_dense)

        outputs = tf.keras.layers.Dense(self.writer_dim, activation='relu')(style_dense)

        self.model = tf.keras.Model(inputs=feature_inputs, outputs=outputs, name=self.name)


class RecognizerModel(tf.keras.Model):
    """
    A recognizer model that transcripts the handwriting text from images.
    """

    def __init__(self,
                 features_shape,
                 lexical_shape,
                 **kwargs):
        """
        Initializes the model class.

        Args:
            features_shape: list or tuple
                Shape of the input features.
            lexical_shape: list or tuple
                Shape of the text sequences and vocabulary encoding.
            **kwargs
                Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.features_shape = features_shape
        self.lexical_shape = lexical_shape

        self.build_model()

    def get_config(self):
        """
        Returns the config of the model.

        Returns:
            A dictionary containing the configuration of the model.
        """

        config = {
            "features_shape": self.features_shape,
            "lexical_shape": self.lexical_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def summary(self,
                line_length=None,
                positions=None,
                print_fn=None,
                expand_nested=False,
                show_trainable=False,
                layer_range=None):
        """
        Prints a string summary of the network.

        Args:
            line_length: int, optional
                Total length of printed lines.
            positions: list of float, optional
                Positions of log elements in each line.
            print_fn: callable, optional
                Function used for printing the summary.
            expand_nested: bool, optional
                Whether to expand the nested models.
            show_trainable: bool, optional
                Whether to show if a layer is trainable.
            layer_range: list or tuple, optional
                Range of layers to include in the model summary.
        """

        self.model.summary(line_length,
                           positions,
                           print_fn,
                           expand_nested,
                           show_trainable,
                           layer_range)

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

        feature_inputs = tf.keras.layers.Input(shape=self.features_shape)

        first_units = tf.math.ceil(
            (self.features_shape[1] * self.lexical_shape[0] * self.lexical_shape[1]) / self.features_shape[0])

        dense = tf.keras.layers.Dense(units=first_units)(feature_inputs)
        dense = tf.keras.layers.Reshape(target_shape=(self.lexical_shape[0]*self.lexical_shape[1], -1))(dense)

        bgru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=True, dropout=0.5))(dense)
        bgru = tf.keras.layers.Dense(units=256)(bgru)

        bgru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=True, dropout=0.5))(bgru)
        bgru = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(bgru)

        outputs = tf.keras.layers.Reshape(target_shape=self.lexical_shape)(bgru)

        self.model = tf.keras.Model(inputs=feature_inputs, outputs=outputs, name=self.name)
