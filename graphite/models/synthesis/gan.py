import tensorflow as tf

from models.components import SpectralSelfAttention
from models.components import ConditionalBatchNormalization
from models.components import SpectralNormalization
from models.components import ExtractPatches
from models.components import CTCLoss
from models.components import CTXLoss


class SynthesisModel(tf.keras.Model):

    def __init__(self,
                 image_shape,
                 patch_shape,
                 lexical_shape,
                 writer_dim,
                 latent_dim,
                 embedding_dim,
                 backbone_blocks,
                 generator_blocks,
                 discriminator_blocks,
                 **kwargs):

        super().__init__(**kwargs)

        self.g_model = GeneratorModel(image_shape=image_shape,
                                      lexical_shape=lexical_shape,
                                      latent_dim=latent_dim,
                                      embedding_dim=embedding_dim,
                                      blocks=generator_blocks,
                                      name='generator')
        # self.g_model.summary()

        self.d_model = DiscriminatorModel(image_shape=image_shape,
                                          patch_shape=None,
                                          lexical_shape=lexical_shape,
                                          embedding_dim=embedding_dim,
                                          blocks=discriminator_blocks,
                                          name='discriminator')
        # self.d_model.summary()

        self.p_model = DiscriminatorModel(image_shape=image_shape,
                                          patch_shape=patch_shape,
                                          lexical_shape=lexical_shape,
                                          embedding_dim=embedding_dim,
                                          blocks=discriminator_blocks,
                                          name='patch_discriminator')
        # self.p_model.summary()

        self.b_model = StyleBackboneModel(image_shape=image_shape,
                                          blocks=backbone_blocks,
                                          name='style_backbone')
        # self.b_model.summary()

        self.e_model = StyleEncoderModel(features_shape=self.b_model.features_shape,
                                         latent_dim=latent_dim,
                                         name='style_encoder')
        # self.e_model.summary()

        self.i_model = IdentifierModel(features_shape=self.b_model.features_shape,
                                       writer_dim=writer_dim,
                                       name='identifier')
        # self.i_model.summary()

        self.r_model = RecognizerModel(image_shape=image_shape,
                                       lexical_shape=lexical_shape,
                                       blocks=backbone_blocks,
                                       name='recognizer')
        # self.r_model.summary()

    def compile(self, learning_rate=0.001):

        super().compile(run_eagerly=False)

        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.b_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.e_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.i_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.r_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.ctx_loss = CTXLoss()
        self.ctc_loss = CTCLoss()
        self.kld_loss = tf.keras.losses.KLDivergence()
        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def train_step(self, data):
        (aug_image_inputs, aug_text_inputs), (image_inputs, text_inputs, writer_inputs) = data

        # # fake_latent + aug_text = fake_full_images
        # latent_shape = (tf.shape(image_inputs)[0], self.e_model.latent_dim)
        # fake_latent_inputs = tf.random.normal(latent_shape, mean=0.0, stddev=1.0)

        with tf.GradientTape(persistent=True) as tape:
            # recognizer loss
            ctc_logits = self.r_model(aug_image_inputs, training=True)
            ctc_loss = self.ctc_loss(text_inputs, ctc_logits)

            # extract filters from backbone
            style_filters, _ = self.b_model(aug_image_inputs, training=True)

            # writer identifier loss
            wid_logits = self.i_model(style_filters, training=True)
            wid_loss = self.cls_loss(writer_inputs, wid_logits)

            # backbone loss
            backbone_loss = wid_loss

            # style encoder
            # style_encoded, _, _ = self.e_model(style_filters, training=True)

        r_gradients = tape.gradient(ctc_loss, self.r_model.trainable_weights)
        self.r_optimizer.apply_gradients(zip(r_gradients, self.r_model.trainable_weights))

        b_gradients = tape.gradient(backbone_loss, self.b_model.trainable_weights)
        self.b_optimizer.apply_gradients(zip(b_gradients, self.b_model.trainable_weights))

        i_gradients = tape.gradient(wid_loss, self.i_model.trainable_weights)
        self.i_optimizer.apply_gradients(zip(i_gradients, self.i_model.trainable_weights))

        return {
            "r_loss": ctc_loss,
            "b_loss": backbone_loss,
            "i_loss": wid_loss,
        }

        # with tf.GradientTape() as d_tape, \
        #         tf.GradientTape() as g_tape, \
        #         tf.GradientTape() as p_tape, \
        #         tf.GradientTape() as b_tape, \
        #         tf.GradientTape() as e_tape, \
        #         tf.GradientTape() as w_tape, \
        #         tf.GradientTape() as r_tape:

        #     # fake_full_images = self.g_model([fake_latent_inputs, aug_text_inputs], training=True)
        #     # g_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_full_images))

        #     # fake_disc = self.d_model(aug_image_inputs, training=True)
        #     # d_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_disc))

        #     # fake_disc = self.p_model(aug_image_inputs, training=True)
        #     # p_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_disc))

        #     fake_disc, _ = self.b_model(aug_image_inputs, training=True)
        #     # b_loss = tf.reduce_mean(fake_disc)

        #     # fake_disc, _, _ = self.e_model(fake_disc, training=True)
        #     # e_loss = tf.reduce_mean(fake_disc)

        #     # fake_disc = self.i_model(fake_disc, training=True)
        #     # w_loss = tf.reduce_mean(fake_disc)

        #     fake_disc = self.r_model(fake_disc, training=True)
        #     r_loss = tf.reduce_mean(fake_disc)

        # # g_grads = g_tape.gradient(g_loss, self.g_model.trainable_weights)
        # # self.g_optimizer.apply_gradients(zip(g_grads, self.g_model.trainable_weights))

        # # d_grads = d_tape.gradient(d_loss, self.d_model.trainable_weights)
        # # self.d_optimizer.apply_gradients(zip(d_grads, self.d_model.trainable_weights))

        # # p_grads = p_tape.gradient(p_loss, self.p_model.trainable_weights)
        # # self.d_optimizer.apply_gradients(zip(p_grads, self.p_model.trainable_weights))

        # # b_grads = b_tape.gradient(b_loss, self.b_model.trainable_weights)
        # # self.d_optimizer.apply_gradients(zip(b_grads, self.b_model.trainable_weights))

        # # e_grads = e_tape.gradient(e_loss, self.b_model.trainable_weights)
        # # self.d_optimizer.apply_gradients(zip(e_grads, self.b_model.trainable_weights))

        # # w_grads = w_tape.gradient(w_loss, self.i_model.trainable_weights)
        # # self.d_optimizer.apply_gradients(zip(w_grads, self.i_model.trainable_weights))

        # r_grads = r_tape.gradient(r_loss, self.r_model.trainable_weights)
        # self.d_optimizer.apply_gradients(zip(r_grads, self.r_model.trainable_weights))

        # return {
        #     # "g_loss": g_loss,
        #     # "d_loss": d_loss,
        #     # "p_loss": p_loss,
        #     # "b_loss": b_loss,
        #     # "e_loss": e_loss,
        #     # "w_loss": w_loss,
        #     "r_loss": r_loss,
        # }
        ####################################

        # # fake_latent + aug_text = fake_full_images
        # latent_shape = (tf.shape(image_inputs)[0], self.e_model.latent_dim)
        # fake_latent_inputs = tf.random.normal(latent_shape, mean=0.0, stddev=1.0)
        # fake_full_images = self.g_model([fake_latent_inputs, aug_text_inputs], training=True)

        # # real_latent + aug_text = fake_partial_images
        # real_features_inputs, _ = self.b_model(image_inputs, training=True)
        # real_latent_inputs, _, _ = self.e_model(real_features_inputs, training=True)
        # fake_partial_images = self.g_model([real_latent_inputs, aug_text_inputs], training=True)

        # # real_latent + real_text = fake_real_images
        # fake_real_images = self.g_model([real_latent_inputs, text_inputs], training=True)

        # # concat and shuffle fake inputs
        # fake_image_inputs = tf.concat([fake_full_images, fake_partial_images, fake_real_images], axis=0)
        # fake_text_inputs = tf.concat([aug_text_inputs, aug_text_inputs, text_inputs], axis=0)

        # indices = tf.range(start=0, limit=tf.shape(fake_image_inputs)[0], dtype=tf.int32)
        # shuffled_indices = tf.random.shuffle(indices)

        # fake_image_inputs = tf.gather(fake_image_inputs, shuffled_indices)
        # fake_text_inputs = tf.gather(fake_text_inputs, shuffled_indices)

        # with tf.GradientTape() as tape:
        #     # discriminator with fake inputs
        #     fake_disc = self.d_model([fake_image_inputs, fake_text_inputs], training=True)
        #     fake_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_disc))

        #     # patch discrimiantor with fake inputs
        #     fake_patch_disc = self.p_model([fake_image_inputs, fake_text_inputs], training=True)
        #     fake_patch_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_patch_disc))

        #     # concat and shuffle real inputs
        #     real_image_inputs = tf.concat([image_inputs, aug_image_inputs], axis=0)
        #     real_text_inputs = tf.concat([text_inputs, aug_text_inputs], axis=0)

        #     indices = tf.range(start=0, limit=tf.shape(real_image_inputs)[0], dtype=tf.int32)
        #     shuffled_indices = tf.random.shuffle(indices)

        #     real_image_inputs = tf.gather(real_image_inputs, shuffled_indices)
        #     real_text_inputs = tf.gather(real_text_inputs, shuffled_indices)

        #     # discriminator with real inputs
        #     real_disc = self.d_model([real_image_inputs, real_text_inputs], training=True)
        #     real_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_disc))

        #     # patch discrimiantor with fake inputs
        #     real_patch_disc = self.p_model([real_image_inputs, real_text_inputs], training=True)
        #     real_patch_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_patch_disc))

        #     # discriminator loss
        #     disc_loss = fake_disc_loss + fake_patch_disc_loss + real_disc_loss + real_patch_disc_loss

        #     # generator...

        # # calculate discriminator gradients and update model weights
        # disc_gradients = tape.gradient(fake_disc_loss, self.d_model.trainable_variables)
        # self.d_model_optimizer.apply_gradients(zip(disc_gradients, self.d_model.trainable_variables))

        # return {"loss": fake_disc_loss}


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

        def residual_block_up(x, y, filters, upsample=None):
            h = ConditionalBatchNormalization()([x, y])
            h = tf.keras.layers.ReLU()(h)

            if upsample is not None:
                h = tf.keras.layers.UpSampling2D(size=upsample, interpolation='nearest')(h)
                x = tf.keras.layers.UpSampling2D(size=upsample, interpolation='nearest')(x)

            h = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(h)

            h = ConditionalBatchNormalization()([h, y])
            h = tf.keras.layers.ReLU()(h)

            h = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(h)
            x = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=1))(x)

            return tf.keras.layers.Add()([h, x])

        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,))

        latent_dense = SpectralNormalization(
            tf.keras.layers.Dense(units=self.latent_dim * len(self.blocks)))(latent_inputs)

        chunks = tf.keras.layers.Lambda(
            lambda x: tf.convert_to_tensor(tf.split(x, len(self.blocks), axis=-1)), name='split')(latent_dense)

        latent_expanded = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=1), name='expand_dims')(latent_inputs)

        text_inputs = tf.keras.layers.Input(shape=self.lexical_shape[:-1])
        text_flattened = tf.keras.layers.Flatten()(text_inputs)

        text_embedding = tf.keras.layers.Embedding(input_dim=self.lexical_shape[-1] + 1,
                                                   output_dim=self.embedding_dim,
                                                   mask_zero=True)(text_flattened)

        latent_tiled = tf.keras.layers.Lambda(
            lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], 1]), name='tile')([latent_expanded, text_embedding])

        latent_text = tf.keras.layers.Concatenate(axis=-1)([latent_tiled, text_embedding])
        latent_text = SpectralNormalization(tf.keras.layers.Dense(units=4 * 4 * 2 * self.blocks[0]))(latent_text)
        latent_text = tf.keras.layers.Reshape(target_shape=(latent_text.get_shape()[1] * 4, 4, -1))(latent_text)

        block = tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 3, 2, 1]), name='transpose')(latent_text)

        for i, filters in enumerate(self.blocks):
            upsample = None

            if i > 0 and (i == len(self.blocks) - 1 or i % 2 == 0):
                block = SpectralSelfAttention()(block)

                height_upsample_required = block.shape[1] < self.image_shape[0]
                width_upsample_required = block.shape[2] < self.image_shape[1]

                if height_upsample_required or width_upsample_required:
                    upsample_height = 2 if height_upsample_required else 1
                    upsample_width = 2 if width_upsample_required else 1
                    upsample = (upsample_height, upsample_width)

            block = residual_block_up(block, chunks[i], filters, upsample=upsample)

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

        def residual_block_down(x, filters, ops_early=True, downsample=False):
            h = x

            if ops_early:
                h = tf.keras.layers.ReLU()(x)

            h = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(h)
            h = tf.keras.layers.ReLU()(h)
            h = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(h)

            if ops_early:
                x = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=1))(x)

            if downsample:
                h = tf.keras.layers.AveragePooling2D()(h)
                x = tf.keras.layers.AveragePooling2D()(x)

            if not ops_early:
                x = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=1))(x)

            return tf.keras.layers.Add()([h, x])

        image_inputs = tf.keras.layers.Input(shape=self.image_shape)

        block = image_inputs if self.patch_shape is None \
            else ExtractPatches(patch_shape=self.patch_shape)(image_inputs)

        for i, filters in enumerate(self.blocks):
            if i > 0 and (i == len(self.blocks) - 1 or i % 2 == 0):
                block = SpectralSelfAttention()(block)

            block = residual_block_down(block, filters, ops_early=(i > 0), downsample=(i < len(self.blocks) - 1))

        outputs = tf.keras.layers.ReLU()(block)
        outputs = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2]))(outputs)
        outputs = SpectralNormalization(tf.keras.layers.Dense(units=1))(outputs)

        self.model = tf.keras.Model(inputs=image_inputs, outputs=outputs, name=self.name)


class StyleBackboneModel(tf.keras.Model):
    """
    A backbone model that extracts style patterns from images.
    """

    def __init__(self,
                 image_shape,
                 blocks,
                 **kwargs):
        """
        Initializes the model class.

        Args:
            image_shape: list or tuple
                Shape of the input image.
            blocks: list or tuple
                Blocks of channels.
            **kwargs
                Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.features_shape = None
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
            "features_shape": self.features_shape,
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

        image_inputs = tf.keras.layers.Input(shape=self.image_shape)

        conv = tf.keras.layers.Conv2D(self.blocks[0], kernel_size=5, strides=2, padding='same')(image_inputs)
        blocks = list(self.blocks) + [self.blocks[-1] * 2]
        feats = []

        for i, filters in enumerate(blocks[:-1]):
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
            block2 = tf.keras.layers.Conv2D(blocks[i + 1], kernel_size=3, strides=1, padding='same')(block2)
            block2 = tf.keras.layers.BatchNormalization()(block2)

            shortcut = tf.keras.layers.Conv2D(blocks[i + 1],
                                              kernel_size=1,
                                              strides=1,
                                              padding='valid',
                                              use_bias=False)(conv)

            conv = tf.keras.layers.Add()([shortcut, block2])
            conv = tf.keras.layers.ZeroPadding2D(padding=1)(conv)
            conv = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(conv)

            feats.append(conv)

        conv = tf.keras.layers.ReLU()(conv)
        conv = tf.keras.layers.Conv2D(blocks[-1], kernel_size=3, strides=1, padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.ReLU()(conv)

        outputs = tf.keras.layers.Reshape(target_shape=(conv.get_shape()[1] * conv.get_shape()[2], -1))(conv)

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

        outputs = tf.keras.layers.Dense(self.writer_dim)(style_dense)

        self.model = tf.keras.Model(inputs=feature_inputs, outputs=outputs, name=self.name)


class RecognizerModel(tf.keras.Model):
    """
    A recognizer model that transcripts the handwriting text from images.
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 blocks,
                 **kwargs):
        """
        Initializes the model class.

        Args:
            image_shape: list or tuple
                Shape of the input image.
            lexical_shape: list or tuple
                Shape of the text sequences and vocabulary encoding.
            blocks: list or tuple
                Blocks of channels.
            **kwargs
                Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
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

        image_inputs = tf.keras.layers.Input(shape=self.image_shape)

        conv = tf.keras.layers.Conv2D(self.blocks[0], kernel_size=5, strides=2, padding='same')(image_inputs)
        blocks = list(self.blocks) + [self.blocks[-1] * 2]

        for i, filters in enumerate(blocks[:-1]):
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
            block2 = tf.keras.layers.Conv2D(blocks[i + 1], kernel_size=3, strides=1, padding='same')(block2)
            block2 = tf.keras.layers.BatchNormalization()(block2)

            shortcut = tf.keras.layers.Conv2D(blocks[i + 1],
                                              kernel_size=1,
                                              strides=1,
                                              padding='valid',
                                              use_bias=False)(conv)

            conv = tf.keras.layers.Add()([shortcut, block2])
            conv = tf.keras.layers.ZeroPadding2D(padding=1)(conv)
            conv = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(conv)

        conv = tf.keras.layers.ReLU()(conv)
        conv = tf.keras.layers.Conv2D(blocks[-1], kernel_size=3, strides=1, padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.ReLU()(conv)

        units = conv.get_shape()[3] * self.lexical_shape[0] * self.lexical_shape[1]
        units = tf.math.ceil(units / (conv.get_shape()[1] * conv.get_shape()[2]))

        dense = tf.keras.layers.Dense(units=units)(conv)
        dense = tf.keras.layers.Reshape(target_shape=(self.lexical_shape[0]*self.lexical_shape[1], -1))(dense)

        bgru = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5))(dense)
        bgru = tf.keras.layers.Dense(units=256)(bgru)

        bgru = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5))(bgru)
        bgru = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(bgru)

        outputs = tf.keras.layers.Reshape(target_shape=self.lexical_shape)(bgru)

        self.model = tf.keras.Model(inputs=image_inputs, outputs=outputs, name=self.name)
