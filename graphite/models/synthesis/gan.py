import tensorflow as tf

from graphite.models.components.layers import ConditionalBatchNormalization
from graphite.models.components.layers import ExtractPatches
from graphite.models.components.layers import SpectralNormalization
from graphite.models.components.layers import SpectralSelfAttention
from graphite.models.components.models import SynthesisBaseModel


class SynthesisModel(SynthesisBaseModel):
    """
    This model integrates several submodels including a generator, discriminator,
        patch discriminator, style backbone, style encoder, writer identification, and text recognition.

    References
    ----------
    GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
        https://arxiv.org/abs/1706.08500

    HiGAN+: Handwriting Imitation GAN with Disentangled Representations
        https://dl.acm.org/doi/10.1145/3550070

    Large Scale GAN Training for High Fidelity Natural Image Synthesis
        https://arxiv.org/abs/1809.11096v2

    Modulating early visual processing by language
        https://arxiv.org/abs/1707.00683v3

    ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation
        https://arxiv.org/abs/2003.10557

    Wasserstein GAN
        https://arxiv.org/abs/1701.07875
    """

    def build_model(self):
        """
        Initializes and builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        latent_dim = 128
        embedding_dim = 128
        patch_shape = [32, 32, 1]
        backbone_blocks = [16, 32, 64, 128]
        generator_blocks = [256, 128, 64, 64]
        discriminator_blocks = [64, 128, 256, 256]

        self.discriminator = DiscriminatorModel(image_shape=self.image_shape,
                                                patch_shape=None,
                                                blocks=discriminator_blocks,
                                                name='discriminator')

        self.patch_discriminator = DiscriminatorModel(image_shape=self.image_shape,
                                                      patch_shape=patch_shape,
                                                      blocks=discriminator_blocks,
                                                      name='patch_discriminator')

        self.style_backbone = BackboneModel(image_shape=self.image_shape,
                                            blocks=backbone_blocks,
                                            name='style_backbone')

        self.identification = IdentificationModel(features_shape=self.style_backbone.features_output_shape,
                                                  writers_shape=self.writers_shape,
                                                  name='identification')

        self.recognition = RecognitionModel(image_shape=self.image_shape,
                                            lexical_shape=self.lexical_shape,
                                            blocks=backbone_blocks,
                                            name='recognition')

        self.style_encoder = StyleEncoderModel(features_shape=self.style_backbone.features_output_shape,
                                               latent_dim=latent_dim,
                                               name='style_encoder')

        self.generator = GeneratorModel(features_shape=self.style_encoder.features_output_shape,
                                        image_shape=self.image_shape,
                                        lexical_shape=self.lexical_shape,
                                        embedding_dim=embedding_dim,
                                        blocks=generator_blocks,
                                        name='generator')

    def train_step(self, input_data):
        """
        Perform the training step on the provided batch of data.

        Parameters
        ----------
        input_data : list or tuple
            A batch of data (x_data, y_data).

        Returns
        -------
        dict
            A dictionary containing metrics and losses.
        """

        (_, aug_text_data, _), (image_data, text_data, writer_data) = input_data

        batch_size = tf.shape(image_data)[0]
        q_batch = tf.math.maximum(1, batch_size // 4)

        # discriminator phase
        for _ in range(2):
            q_indices = tf.random.shuffle(tf.range(batch_size))[:q_batch]
            q_image_data = tf.gather(image_data, q_indices)
            q_text_data = tf.gather(text_data, q_indices)

            q_aug_indices = tf.random.shuffle(tf.range(batch_size))[:q_batch]
            q_aug_text_data = tf.gather(aug_text_data, q_aug_indices)

            fake_latent_data = tf.random.normal((q_batch, self.style_encoder.latent_dim))

            real_features_data, _ = self.style_backbone(q_image_data, training=True)
            real_latent_data, _, _ = self.style_encoder(real_features_data, training=True)

            fake_fake_images = self.generator([fake_latent_data, q_aug_text_data], training=True)
            real_fake_images = self.generator([real_latent_data, q_aug_text_data], training=True)
            real_real_images = self.generator([real_latent_data, q_text_data], training=True)
            fake_real_images = self.generator([fake_latent_data, q_text_data], training=True)

            fake_image_data = tf.random.shuffle(tf.concat([fake_fake_images,
                                                           real_fake_images,
                                                           real_real_images,
                                                           fake_real_images], axis=0))
            # patch and discriminator loss
            with tf.GradientTape() as p_tape, tf.GradientTape() as d_tape:
                fake_patch_disc = self.patch_discriminator(fake_image_data, training=True)
                fake_patch_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_patch_disc))

                real_patch_disc = self.patch_discriminator(image_data, training=True)
                real_patch_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_patch_disc))

                fake_disc = self.discriminator(fake_image_data, training=True)
                fake_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_disc))

                real_disc = self.discriminator(image_data, training=True)
                real_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_disc))

                p_loss = fake_patch_disc_loss + real_patch_disc_loss
                d_loss = fake_disc_loss + fake_patch_disc_loss + real_disc_loss + real_patch_disc_loss

            d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_weights))

            p_gradients = p_tape.gradient(p_loss, self.patch_discriminator.trainable_weights)
            self.p_optimizer.apply_gradients(zip(p_gradients, self.patch_discriminator.trainable_weights))

            # writer identification and recognition loss
            with tf.GradientTape() as b_tape, \
                    tf.GradientTape() as w_tape, \
                    tf.GradientTape() as r_tape:

                features_data, _ = self.style_backbone(image_data, training=True)
                wid_logits = self.identification(features_data, training=True)
                w_loss = self.cls_loss(writer_data, wid_logits)

                ctc_logits = self.recognition(image_data, training=True)
                r_loss = self.ctc_loss(text_data, ctc_logits)

            b_gradients = b_tape.gradient(w_loss, self.style_backbone.trainable_weights)
            self.b_optimizer.apply_gradients(zip(b_gradients, self.style_backbone.trainable_weights))

            w_gradients = w_tape.gradient(w_loss, self.identification.trainable_weights)
            self.w_optimizer.apply_gradients(zip(w_gradients, self.identification.trainable_weights))

            r_gradients = r_tape.gradient(r_loss, self.recognition.trainable_weights)
            self.r_optimizer.apply_gradients(zip(r_gradients, self.recognition.trainable_weights))

        # generator phase
        indices = tf.random.shuffle(tf.range(batch_size))

        q_image_data = tf.gather(image_data, indices[:q_batch])
        q_text_data = tf.gather(text_data, indices[:q_batch])
        q_aug_text_data = tf.gather(aug_text_data, indices[:q_batch])
        q_writer_data = tf.gather(writer_data, indices[:q_batch])

        fake_latent_data = tf.random.normal((q_batch, self.style_encoder.latent_dim))
        real_features_data, real_image_feats = self.style_backbone(q_image_data, training=True)

        with tf.GradientTape() as e_tape, tf.GradientTape() as g_tape:
            real_latent_data, mu, logvar = self.style_encoder(real_features_data, training=True)

            fake_fake_images = self.generator([fake_latent_data, q_aug_text_data], training=True)
            real_fake_images = self.generator([real_latent_data, q_aug_text_data], training=True)
            real_real_images = self.generator([real_latent_data, q_text_data], training=True)
            fake_real_images = self.generator([fake_latent_data, q_text_data], training=True)

            # kl-divergency loss
            kl_loss = self.kld_loss(mu, logvar)

            # style reconstruction loss
            with e_tape.stop_recording(), g_tape.stop_recording():
                fake_fake_features_data, _ = self.style_backbone(fake_fake_images, training=True)
                fake_real_features_data, _ = self.style_backbone(fake_real_images, training=True)

            fake_fake_latent_data, _, _ = self.style_encoder(fake_fake_features_data, training=True)
            fake_fake_info_loss = tf.reduce_mean(tf.math.abs(fake_fake_latent_data - fake_latent_data))

            fake_real_latent_data, _, _ = self.style_encoder(fake_real_features_data, training=True)
            fake_real_info_loss = tf.reduce_mean(tf.math.abs(fake_real_latent_data - fake_latent_data))

            info_loss = tf.reduce_mean([fake_fake_info_loss, fake_real_info_loss])

            # content restruction loss
            real_real_l1_loss = self.l1_loss(q_image_data, real_real_images)
            l1_loss = tf.reduce_mean(real_real_l1_loss)

            # patch and discriminator loss
            with e_tape.stop_recording(), g_tape.stop_recording():
                fake_image_data = tf.random.shuffle(tf.concat([fake_fake_images,
                                                               real_fake_images,
                                                               real_real_images,
                                                               fake_real_images], axis=0))

                fake_disc = self.discriminator(fake_image_data, training=False)
                fake_patch_disc = self.patch_discriminator(fake_image_data, training=False)

            fake_disc_loss = -tf.reduce_mean(fake_disc)
            fake_patch_disc_loss = -tf.reduce_mean(fake_patch_disc)

            disc_loss = fake_disc_loss + fake_patch_disc_loss

            # ctc loss
            with e_tape.stop_recording(), g_tape.stop_recording():
                fake_fake_image_data = tf.random.shuffle(tf.concat([fake_fake_images, real_fake_images], axis=0))
                fake_fake_text_data = tf.concat([q_aug_text_data, q_aug_text_data], axis=0)
                fake_fake_ctc_logits = self.recognition(fake_fake_image_data, training=False)

                fake_real_image_data = tf.random.shuffle(tf.concat([real_real_images, fake_real_images], axis=0))
                fake_real_text_data = tf.concat([q_text_data, q_text_data], axis=0)
                fake_real_ctc_logits = self.recognition(fake_real_image_data, training=False)

            fake_fake_ctc_loss = self.ctc_loss(fake_fake_text_data, fake_fake_ctc_logits)
            fake_real_ctc_loss = self.ctc_loss(fake_real_text_data, fake_real_ctc_logits)

            ctc_loss = fake_fake_ctc_loss + fake_real_ctc_loss

            # writer identify loss
            with e_tape.stop_recording(), g_tape.stop_recording():
                real_fake_features_data, real_fake_image_feats = self.style_backbone(real_fake_images, training=False)
                real_fake_wid_logits = self.identification(real_fake_features_data, training=False)

                real_real_features_data, real_real_image_feats = self.style_backbone(real_real_images, training=False)
                real_real_wid_logits = self.identification(real_real_features_data, training=False)

            real_fake_wid_loss = self.cls_loss(q_writer_data, real_fake_wid_logits)
            real_real_wid_loss = self.cls_loss(q_writer_data, real_real_wid_logits)

            wid_loss = tf.reduce_mean([real_fake_wid_loss, real_real_wid_loss])

            # contextual loss
            ctx_loss = tf.constant(0.0)

            for real_image_feat, real_fake_image_feat, real_real_image_feat \
                    in zip(real_image_feats, real_fake_image_feats, real_real_image_feats):
                ctx_loss += self.ctx_loss(real_image_feat, real_fake_image_feat)
                ctx_loss += self.ctx_loss(real_image_feat, real_real_image_feat)

            # generator loss
            g_loss = (kl_loss * 1e-4) + info_loss + l1_loss + disc_loss + ctc_loss + wid_loss + (ctx_loss * 5.0)

        e_gradients = e_tape.gradient(g_loss, self.style_encoder.trainable_weights)
        self.e_optimizer.apply_gradients(zip(e_gradients, self.style_encoder.trainable_weights))

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_weights))

        # metric phase
        features_data, _ = self.style_backbone(image_data, training=False)
        latent_data, _, _ = self.style_encoder(features_data, training=False)
        generated_images = self.generator([latent_data, text_data], training=False)

        self.kid.update_state(image_data, generated_images)

        return {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'p_loss': p_loss,
            'w_loss': w_loss,
            'r_loss': r_loss,
            self.kid.name: self.kid.result(),
        }


class GeneratorModel(tf.keras.Model):
    """
    A generator model that combines latent and vocabulary data for generative tasks.

    This model synthesizes images based on the combination of latent space vectors and
        lexical information, suitable for tasks like image generation from textual descriptions.
    """

    def __init__(self,
                 features_shape,
                 image_shape,
                 lexical_shape,
                 embedding_dim,
                 blocks,
                 **kwargs):
        """
        Initialize the generator model with specified parameters.

        Parameters
        ----------
        features_shape : list or tuple
            Shape of the input features.
        image_shape : list or tuple
            Shape of the output image.
        lexical_shape : list or tuple
            Shape of the text sequences and vocabulary encoding.
        embedding_dim : int
            Dimension of the embedding space.
        blocks : list or tuple
            Blocks of channels for the model's architecture.
        **kwargs : dict
            Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.features_shape = features_shape
        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.embedding_dim = embedding_dim
        self.blocks = blocks

        self.build_model()

        if hasattr(self, 'model'):
            self.summary = self.model.summary
            self.call = self.model.call

    def get_config(self):
        """
        Retrieves the configuration of the model for serialization.

        Returns
        -------
        dict
            A dictionary containing the configuration of the model.
        """

        config = super().get_config()

        config.update({
            'features_shape': self.features_shape,
            'image_shape': self.image_shape,
            'lexical_shape': self.lexical_shape,
            'embedding_dim': self.embedding_dim,
            'blocks': self.blocks,
        })

        return config

    def build_model(self):
        """
        Initializes and builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
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

        latent_inputs = tf.keras.layers.Input(shape=self.features_shape)

        latent_dense = SpectralNormalization(
            tf.keras.layers.Dense(units=self.features_shape[0] * len(self.blocks)))(latent_inputs)

        chunks = tf.keras.layers.Lambda(
            lambda x: tf.convert_to_tensor(tf.split(x, len(self.blocks), axis=-1)), name='split')(latent_dense)

        latent_expanded = tf. keras.layers.Lambda(
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
            if i == 1:
                block = SpectralSelfAttention()(block)

            upsample = None
            if i > 0 and (i % 2 == 0 or i == len(self.blocks) - 1):
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

    This model is designed to distinguish between real and synthesized images,
        typically used in GAN architectures for generative tasks.
    """

    def __init__(self,
                 image_shape,
                 patch_shape,
                 blocks,
                 **kwargs):
        """
        Initialize the discriminator model with specified parameters.

        Parameters
        ----------
        image_shape : list or tuple
            Shape of the input image.
        patch_shape : list, tuple or None
            Defines whether to apply patches for processing.
        blocks : list or tuple
            Blocks of channels for the model's architecture.
        **kwargs : dict
            Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.blocks = blocks

        self.build_model()

        if hasattr(self, 'model'):
            self.summary = self.model.summary
            self.call = self.model.call

    def get_config(self):
        """
        Retrieves the configuration of the model for serialization.

        Returns
        -------
        dict
            A dictionary containing the configuration of the model.
        """

        config = super().get_config()

        config.update({
            'image_shape': self.image_shape,
            'patch_shape': self.patch_shape,
            'blocks': self.blocks,
        })

    def build_model(self):
        """
        Initializes and builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        def residual_block_down(x, filters, preactive=True, downsample=False):
            h = x

            if preactive:
                h = tf.keras.layers.ReLU()(x)

            h = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(h)
            h = tf.keras.layers.ReLU()(h)
            h = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(h)

            if preactive:
                x = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=1))(x)

            if downsample:
                h = tf.keras.layers.AveragePooling2D()(h)
                x = tf.keras.layers.AveragePooling2D()(x)

            if not preactive:
                x = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=1))(x)

            return tf.keras.layers.Add()([h, x])

        image_inputs = tf.keras.layers.Input(shape=self.image_shape)
        block = ExtractPatches(patch_shape=self.patch_shape or self.image_shape)(image_inputs)

        for i, filters in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                block = SpectralSelfAttention()(block)

            block = residual_block_down(block, filters, preactive=(i > 0), downsample=(i < len(self.blocks) - 1))

        outputs = tf.keras.layers.ReLU()(block)
        outputs = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2]), name='reduce')(outputs)
        outputs = SpectralNormalization(tf.keras.layers.Dense(units=1))(outputs)

        self.model = tf.keras.Model(inputs=image_inputs, outputs=outputs, name=self.name)


class BackboneModel(tf.keras.Model):
    """
    A backbone model that extracts style patterns from images.

    This model is designed to capture the characteristics from input images,
        providing a basis for further style-related processing in generative tasks.
    """

    def __init__(self,
                 image_shape,
                 blocks,
                 **kwargs):
        """
        Initialize the backbone model with specified parameters.

        Parameters
        ----------
        image_shape : list or tuple
            Shape of the input image.
        blocks : list or tuple
            Blocks of channels for the model's architecture.
        **kwargs : dict
            Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.features_shape = None
        self.blocks = blocks

        self.build_model()

        if hasattr(self, 'model'):
            self.summary = self.model.summary
            self.call = self.model.call

    def get_config(self):
        """
        Retrieves the configuration of the model for serialization.

        Returns
        -------
        dict
            A dictionary containing the configuration of the model.
        """

        config = super().get_config()

        config.update({
            'image_shape': self.image_shape,
            'features_shape': self.features_shape,
            'blocks': self.blocks,
        })

    def build_model(self):
        """
        Initializes and builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
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

            strides = (2, 2) if i + 1 == len(blocks[:-1]) // 2 else (1, 2)
            conv = tf.keras.layers.MaxPool2D(pool_size=3, strides=strides)(conv)

            feats.append(conv)

        conv = tf.keras.layers.ReLU()(conv)
        conv = tf.keras.layers.Conv2D(blocks[-1], kernel_size=3, strides=1, padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.ReLU()(conv)

        outputs = tf.keras.layers.Reshape(target_shape=(-1, conv.get_shape()[-1]))(conv)

        self.features_output_shape = outputs.get_shape()[1:]
        self.model = tf.keras.Model(inputs=image_inputs, outputs=[outputs, feats], name=self.name)


class StyleEncoderModel(tf.keras.Model):
    """
    An encoder model that encodes extracted style features from images into a representative style vector.

    This model is part of a generative architecture where style features are encoded into a latent
        representation, facilitating the generation of images with specific stylistic attributes.
    """

    def __init__(self,
                 features_shape,
                 latent_dim,
                 **kwargs):
        """
        Initialize the style encoder model with specified parameters.

        Parameters
        ----------
        features_shape : list or tuple
            Shape of the input features.
        latent_dim : int
            Dimension of the latent space.
        **kwargs : dict
            Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.features_shape = features_shape
        self.latent_dim = latent_dim

        self.build_model()

        if hasattr(self, 'model'):
            self.summary = self.model.summary
            self.call = self.model.call

    def get_config(self):
        """
        Retrieves the configuration of the model for serialization.

        Returns
        -------
        dict
            A dictionary containing the configuration of the model.
        """

        config = super().get_config()

        config.update({
            'features_shape': self.features_shape,
            'latent_dim': self.latent_dim,
        })

    def build_model(self):
        """
        Initializes and builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        feature_inputs = tf.keras.layers.Input(shape=self.features_shape)

        style = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=-2) / tf.cast(tf.shape(x)[-2], tf.float32) + 1e-7,
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

        self.features_output_shape = outputs.get_shape()[1:]
        self.model = tf.keras.Model(inputs=feature_inputs, outputs=[outputs, mu, logvar], name=self.name)


class IdentificationModel(tf.keras.Model):
    """
    A writer identification model that classifies handwriting images based on extracted style features.

    This model is designed to identify the writer of a given handwriting sample
        by analyzing the stylistic features of the handwriting.
    """

    def __init__(self,
                 features_shape,
                 writers_shape,
                 **kwargs):
        """
        Initialize the writer identification model with specified parameters.

        Parameters
        ----------
        features_shape : list or tuple
            Shape of the input features.
        writers_shape : int
            Number of writers to classify.
        **kwargs : dict
            Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.features_shape = features_shape
        self.writers_shape = writers_shape

        self.build_model()

        if hasattr(self, 'model'):
            self.summary = self.model.summary
            self.call = self.model.call

    def get_config(self):
        """
        Retrieves the configuration of the model for serialization.

        Returns
        -------
        dict
            A dictionary containing the configuration of the model.
        """

        config = super().get_config()

        config.update({
            'features_shape': self.features_shape,
            'writers_shape': self.writers_shape,
        })

    def build_model(self):
        """
        Initializes and builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        feature_inputs = tf.keras.layers.Input(shape=self.features_shape)

        style = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=-2) / tf.cast(tf.shape(x)[-2], tf.float32) + 1e-7,
            name='reduce')(feature_inputs)

        style_dense = tf.keras.layers.Dense(self.features_shape[-1])(style)
        style_dense = tf.keras.layers.LeakyReLU(alpha=0.01)(style_dense)

        outputs = tf.keras.layers.Dense(self.writers_shape[0])(style_dense)

        self.model = tf.keras.Model(inputs=feature_inputs, outputs=outputs, name=self.name)


class RecognitionModel(tf.keras.Model):
    """
    A recognition model that transcribes handwritten texts from images.

    This model is designed to extract textual information from images of handwriting,
        facilitating tasks like optical character recognition and handwriting analysis.
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 blocks,
                 **kwargs):
        """
        Initialize the handwriting recognition model with specified parameters.

        Parameters
        ----------
        image_shape : list or tuple
            Shape of the image input.
        lexical_shape : list or tuple
            Shape of the text sequences and vocabulary encoding.
        blocks : list or tuple
            Blocks of channels for the model's architecture.
        **kwargs : dict
            Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.blocks = blocks

        self.build_model()

        if hasattr(self, 'model'):
            self.summary = self.model.summary
            self.call = self.model.call

    def get_config(self):
        """
        Retrieves the configuration of the model for serialization.

        Returns
        -------
        dict
            A dictionary containing the configuration of the model.
        """

        config = super().get_config()

        config.update({
            'image_shape': self.image_shape,
            'lexical_shape': self.lexical_shape,
            'blocks': self.blocks,
        })

    def build_model(self):
        """
        Initializes and builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
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

            strides = (2, 2) if i + 1 == len(blocks[:-1]) // 2 else (1, 2)
            conv = tf.keras.layers.MaxPool2D(pool_size=3, strides=strides)(conv)

        conv = tf.keras.layers.ReLU()(conv)
        conv = tf.keras.layers.Conv2D(blocks[-1], kernel_size=3, strides=1, padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.ReLU()(conv)

        bgru = tf.keras.layers.Reshape(target_shape=(conv.get_shape()[1], -1))(conv)

        bgru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.5))(bgru)
        bgru = tf.keras.layers.Dense(units=256)(bgru)

        bgru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.5))(bgru)
        bgru = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(bgru)

        outputs = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), name='expand_dims')(bgru)

        self.model = tf.keras.Model(inputs=image_inputs, outputs=outputs, name=self.name)
