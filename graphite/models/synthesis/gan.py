import tensorflow as tf

from graphite.models.components.common import BaseModel
from graphite.models.components.common import BaseSynthesisModel
from graphite.models.components.common import MetricsTracker
from graphite.models.components.layers import ConditionalBatchNormalization
from graphite.models.components.layers import ExtractPatches
from graphite.models.components.layers import MaskingPadding
from graphite.models.components.layers import SelfAttention
from graphite.models.components.layers import SpectralNormalization
from graphite.models.components.optimizers import NormalizedOptimizer
from graphite.models.recognition.bluche import RecognitionModel as RecognitionModel2


class SynthesisModel(BaseSynthesisModel):
    """
    This model integrates several submodels including a generator, discriminator,
        patch discriminator, style backbone, style encoder, writer identification, and text recognition.

    References
    ----------
    Adversarial Generation of Handwritten Text Images Conditioned on Sequences
        https://arxiv.org/abs/1903.00277

    GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium
        https://arxiv.org/abs/1706.08500

    Geometric GAN
        https://arxiv.org/abs/1705.02894

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

    def compile(self, learning_rate=None):
        """
        Compiles neural network model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        """

        super().compile(run_eagerly=False)

        if learning_rate is None:
            learning_rate = 2e-4

        self.d_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999))

        self.g_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999))

        self.w_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999))

        self.r_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999))

    def build_model(self):
        """
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        latent_dim = 128
        embedding_dim = 32
        patch_shape = [32, 32, 1]
        backbone_blocks = [16, 32, 64, 128]
        discriminator_blocks = [64, 128, 256, 256]
        generator_blocks = [256, 128, 64, 64]

        self.discriminator = DiscriminatorModel(image_shape=self.image_shape,
                                                blocks=discriminator_blocks,
                                                name='discriminator')

        self.patch_discriminator = DiscriminatorModel(image_shape=self.image_shape,
                                                      blocks=discriminator_blocks,
                                                      patch_shape=patch_shape,
                                                      name='patch_discriminator')

        self.style_backbone = BackboneModel(image_shape=self.image_shape,
                                            blocks=backbone_blocks,
                                            name='style_backbone')

        self.identification = IdentificationModel(features_shape=self.style_backbone.features_output_shape,
                                                  writers_shape=self.writers_shape,
                                                  name='identification')

        # self.recognition = RecognitionModel(image_shape=self.image_shape,
        #                                     lexical_shape=self.lexical_shape,
        #                                     blocks=backbone_blocks,
        #                                     name='recognition')

        self.recognition = RecognitionModel2(image_shape=self.image_shape,
                                             lexical_shape=self.lexical_shape,
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

        self.metrics_tracker = MetricsTracker()

    def _discriminator_step(self, input_data):
        """
        Update the discriminator model using the provided batch of input data.

        Parameters
        ----------
        input_data : list or tuple
            A batch of data (x_data, y_data).
        """

        (aug_image_data, aug_text_data), (image_data, text_data, writer_data) = input_data

        batch_size = tf.shape(image_data)[0]

        self.discriminator.trainable = True
        self.patch_discriminator.trainable = True
        self.style_backbone.trainable = True
        self.identification.trainable = True
        self.recognition.trainable = True
        self.style_encoder.trainable = False
        self.generator.trainable = False

        for _ in range(self.discriminator_steps):
            real_features_data, _ = self.style_backbone(image_data, training=True)
            real_latent_data, _, _ = self.style_encoder(real_features_data, training=True)

            real_s_real_t_images = self.generator([real_latent_data, text_data], training=True)
            real_s_fake_t_images = self.generator([real_latent_data, aug_text_data], training=True)

            random_latent_data = tf.stop_gradient(tf.random.normal((batch_size, self.style_encoder.latent_dim)))
            fake_s_fake_t_images = self.generator([random_latent_data, aug_text_data], training=True)

            real_s_real_t_images = tf.stop_gradient(real_s_real_t_images)
            real_s_fake_t_images = tf.stop_gradient(real_s_fake_t_images)
            fake_s_fake_t_images = tf.stop_gradient(fake_s_fake_t_images)

            # patch and discriminator
            with tf.GradientTape() as tape:
                # fake images
                real_s_real_t_disc = self.discriminator(real_s_real_t_images, training=True)
                real_s_fake_t_disc = self.discriminator(real_s_fake_t_images, training=True)
                fake_s_fake_t_disc = self.discriminator(fake_s_fake_t_images, training=True)

                fake_disc = tf.concat([real_s_real_t_disc,
                                       real_s_fake_t_disc,
                                       fake_s_fake_t_disc], axis=0)
                fake_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_disc))

                real_s_real_t_patch_disc = self.patch_discriminator(real_s_real_t_images, training=True)
                real_s_fake_t_patch_disc = self.patch_discriminator(real_s_fake_t_images, training=True)
                fake_s_fake_t_patch_disc = self.patch_discriminator(fake_s_fake_t_images, training=True)

                fake_patch_disc = tf.concat([real_s_real_t_patch_disc,
                                             real_s_fake_t_patch_disc,
                                             fake_s_fake_t_patch_disc], axis=0)
                fake_patch_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_patch_disc))

                # real images
                real_disc = self.discriminator(image_data, training=True)
                real_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_disc))

                real_patch_disc = self.patch_discriminator(image_data, training=True)
                real_patch_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_patch_disc))

                # discriminator loss
                d_disc_loss = fake_disc_loss + fake_patch_disc_loss + real_disc_loss + real_patch_disc_loss

            d_gradients = tape.gradient(d_disc_loss, self.discriminator.trainable_variables +
                                        self.patch_discriminator.trainable_weights)

            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables +
                                                 self.patch_discriminator.trainable_weights))

            # writer identifier
            with tf.GradientTape() as tape:
                wid_features_data, _ = self.style_backbone(aug_image_data, training=True)
                wid_logits = self.identification(wid_features_data, training=True)
                d_wid_loss = self.cls_loss(writer_data, wid_logits)

            w_gradients = tape.gradient(d_wid_loss, self.style_backbone.trainable_weights +
                                        self.identification.trainable_weights)

            self.w_optimizer.apply_gradients(zip(w_gradients, self.style_backbone.trainable_weights +
                                                 self.identification.trainable_weights))

            # handwriting recognition
            with tf.GradientTape() as tape:
                ctc_logits = self.recognition(aug_image_data, training=True)
                d_ctc_loss = self.ctc_loss(text_data, ctc_logits)

            r_gradients = tape.gradient(d_ctc_loss, self.recognition.trainable_weights)
            self.r_optimizer.apply_gradients(zip(r_gradients, self.recognition.trainable_weights))

        self.metrics_tracker.update({
            'd_disc_loss': d_disc_loss,
            'd_wid_loss': d_wid_loss,
            'd_ctc_loss': d_ctc_loss,
        })

    def _generator_step(self, input_data):
        """
        Update the generator model using the provided batch of input data.

        Parameters
        ----------
        input_data : list or tuple
            A batch of data (x_data, y_data).
        """

        (_, aug_text_data), (image_data, text_data, writer_data) = input_data

        batch_size = tf.shape(image_data)[0]

        self.discriminator.trainable = False
        self.patch_discriminator.trainable = False
        self.style_backbone.trainable = False
        self.identification.trainable = False
        self.recognition.trainable = False
        self.style_encoder.trainable = True
        self.generator.trainable = True

        with tf.GradientTape() as tape:
            real_features_data, real_image_feats = self.style_backbone(image_data, training=True)

            real_latent_data, mu, logvar = self.style_encoder(real_features_data, training=True)

            real_s_real_t_images = self.generator([real_latent_data, text_data], training=True)
            real_s_fake_t_images = self.generator([real_latent_data, aug_text_data], training=True)

            random_latent_data = tf.stop_gradient(tf.random.normal((batch_size, self.style_encoder.latent_dim)))
            fake_s_fake_t_images = self.generator([random_latent_data, aug_text_data], training=True)

            # patch and discriminator (adversarial)
            real_s_real_t_adv = self.discriminator(real_s_real_t_images, training=True)
            real_s_fake_t_adv = self.discriminator(real_s_fake_t_images, training=True)
            fake_s_fake_t_adv = self.discriminator(fake_s_fake_t_images, training=True)

            fake_adv_disc = tf.concat([real_s_real_t_adv,
                                       real_s_fake_t_adv,
                                       fake_s_fake_t_adv], axis=0)
            fake_adv_loss = -tf.reduce_mean(fake_adv_disc)

            real_s_real_t_patch_adv = self.patch_discriminator(real_s_real_t_images, training=True)
            real_s_fake_t_patch_adv = self.patch_discriminator(real_s_fake_t_images, training=True)
            fake_s_fake_t_patch_adv = self.patch_discriminator(fake_s_fake_t_images, training=True)

            fake_patch_adv_disc = tf.concat([real_s_real_t_patch_adv,
                                             real_s_fake_t_patch_adv,
                                             fake_s_fake_t_patch_adv], axis=0)
            fake_patch_adv_loss = -tf.reduce_mean(fake_patch_adv_disc)

            g_disc_loss = fake_adv_loss + fake_patch_adv_loss

            # handwriting recognition
            real_s_real_t_ctc = self.recognition(real_s_real_t_images, training=True)
            real_s_real_t_ctc_loss = self.ctc_loss(text_data, real_s_real_t_ctc)

            real_s_fake_t_ctc = self.recognition(real_s_fake_t_images, training=True)
            real_s_fake_t_ctc_loss = self.ctc_loss(aug_text_data, real_s_fake_t_ctc)

            fake_s_fake_t_ctc = self.recognition(fake_s_fake_t_images, training=True)
            fake_s_fake_t_ctc_loss = self.ctc_loss(aug_text_data, fake_s_fake_t_ctc)

            g_ctc_loss = real_s_real_t_ctc_loss + real_s_fake_t_ctc_loss + fake_s_fake_t_ctc_loss

            # style reconstruction
            fake_features_data, _ = self.style_backbone(fake_s_fake_t_images, training=True)
            fake_latent_data, _, _ = self.style_encoder(fake_features_data, training=True)

            g_sty_loss = tf.reduce_mean(tf.math.abs(fake_latent_data - random_latent_data))

            # content reconstruction
            g_cnt_loss = self.bv_loss(image_data, (real_s_real_t_images, real_latent_data, mu, logvar))

            # writer identifier
            real_t_features_data, real_t_image_feats = self.style_backbone(real_s_real_t_images, training=True)
            real_t_wid_logits = self.identification(real_t_features_data, training=True)

            fake_t_features_data, fake_t_image_feats = self.style_backbone(real_s_fake_t_images, training=True)
            fake_t_wid_logits = self.identification(fake_t_features_data, training=True)

            real_fake_t_wid_logits = tf.concat([real_t_wid_logits, fake_t_wid_logits], axis=0)
            g_wid_loss = self.cls_loss(tf.repeat(writer_data, repeats=2, axis=0), real_fake_t_wid_logits)

            # contextual
            g_cx_loss = tf.constant(0.0)

            for real_image_feat, real_t_image_feat, fake_t_image_feat in \
                    zip(real_image_feats, real_t_image_feats, fake_t_image_feats):

                g_cx_loss += self.cx_loss(real_image_feat, real_t_image_feat)
                g_cx_loss += self.cx_loss(real_image_feat, fake_t_image_feat)

            # generator loss
            g_loss = g_disc_loss + g_ctc_loss + g_sty_loss + g_cnt_loss + g_wid_loss + (g_cx_loss * 2.0)

        g_gradients = tape.gradient(g_loss, self.style_encoder.trainable_weights +
                                    self.generator.trainable_weights)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.style_encoder.trainable_weights +
                                             self.generator.trainable_weights))

        self.metrics_tracker.update({
            'g_disc_loss': g_disc_loss,
            'g_wid_loss': g_wid_loss,
            'g_ctc_loss': g_ctc_loss,
            'g_sty_loss': g_sty_loss,
            'g_cnt_loss': g_cnt_loss,
            'g_cx_loss': g_cx_loss,
            'g_loss': g_loss,
        })

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

        tf.cond(pred=tf.math.equal(tf.math.mod(self.global_steps, self.generator_steps), 0),
                true_fn=lambda: (self._discriminator_step(input_data), self._generator_step(input_data)),
                false_fn=lambda: (self._discriminator_step(input_data), None))

        self.global_steps.assign_add(delta=1)

        # kid metric
        _, (image_data, text_data, _) = input_data

        features_data, _ = self.style_backbone(image_data, training=False)
        latent_data, _, _ = self.style_encoder(features_data, training=False)
        generated_images = self.generator([latent_data, text_data], training=False)

        self.kid.update_state(image_data, generated_images)
        self.metrics_tracker.update({self.kid.name: self.kid.result()})

        return self.metrics_tracker.result()


class GeneratorModel(BaseModel):
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
                 name='generator',
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
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name=name, **kwargs)

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
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        def residual_block_up(x, y, filters, upsample=None):
            h = ConditionalBatchNormalization()([x, y])
            h = tf.keras.layers.LeakyReLU(alpha=0.01)(h)

            # if upsample:
            #     h = SpectralNormalization(
            #         tf.keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=upsample, padding='same'))(h)
            #     x = SpectralNormalization(
            #         tf.keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=upsample, padding='same'))(x)

            if upsample:
                h = tf.keras.layers.UpSampling2D(size=upsample)(h)
                x = tf.keras.layers.UpSampling2D(size=upsample)(x)

            h = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(h)
            h = ConditionalBatchNormalization()([h, y])
            h = tf.keras.layers.LeakyReLU(alpha=0.01)(h)

            h = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(h)
            x = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=1))(x)

            return tf.keras.layers.Add()([h, x])

        latent_inputs = tf.keras.layers.Input(shape=self.features_shape)

        text_inputs = tf.keras.layers.Input(shape=self.lexical_shape[:-1])
        text_flattened = tf.keras.layers.Flatten()(text_inputs)

        latent_dense = SpectralNormalization(
            tf.keras.layers.Dense(units=self.features_shape[0] * len(self.blocks)))(latent_inputs)

        latent_chunks = tf.keras.layers.Lambda(
            lambda x: tf.transpose(
                tf.split(x, self.features_shape[0], axis=1), perm=[2, 1, 0]), name='chunks')(latent_dense)

        text_embedding = tf.keras.layers.Embedding(input_dim=self.lexical_shape[-1] + 1,
                                                   output_dim=self.embedding_dim,
                                                   mask_zero=True)(text_flattened)

        latent_expanded = tf. keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=1), name='expand')(latent_inputs)

        latent_tiled = tf.keras.layers.Lambda(
            lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], 1]), name='tile')([latent_expanded, text_embedding])

        latent_text = tf.keras.layers.Concatenate(axis=-1)([latent_tiled, text_embedding])
        latent_text = SpectralNormalization(tf.keras.layers.Dense(units=4 * 4 * 2 * self.blocks[0]))(latent_text)

        block = tf.keras.layers.Reshape(target_shape=(latent_text.get_shape()[1] * 4, 4, -1))(latent_text)

        for i, filters in enumerate(self.blocks):
            upsample = None

            height_upsample_required = i > 0 and block.shape[1] < self.image_shape[0]
            width_upsample_required = block.shape[2] < self.image_shape[1]

            if height_upsample_required or width_upsample_required:
                upsample_height = 2 if height_upsample_required else 1
                upsample_width = 2 if width_upsample_required else 1
                upsample = (upsample_height, upsample_width)

            block = residual_block_up(block, latent_chunks[i], filters, upsample=upsample)

            # if i == 0:
            #     block = SelfAttention(spectral_norm=True)(block)

        outputs = tf.keras.layers.BatchNormalization(renorm=True)(block)
        outputs = tf.keras.layers.LeakyReLU(alpha=0.01)(outputs)

        outputs = SpectralNormalization(
            tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', activation='tanh'))(outputs)

        self.model = tf.keras.Model(inputs=[latent_inputs, text_inputs], outputs=outputs, name=self.name)


class DiscriminatorModel(BaseModel):
    """
    A discriminator model that evaluates the authenticity of generated images.

    This model is designed to distinguish between real and synthesized images,
        typically used in GAN architectures for generative tasks.
    """

    def __init__(self,
                 image_shape,
                 blocks,
                 patch_shape=None,
                 name='discriminator',
                 **kwargs):
        """
        Initialize the discriminator model with specified parameters.

        Parameters
        ----------
        image_shape : list or tuple
            Shape of the input image.
        blocks : list or tuple
            Blocks of channels for the model's architecture.
        patch_shape : list, tuple or None
            Patch shape values.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name=name, **kwargs)

        self.image_shape = image_shape
        self.blocks = blocks
        self.patch_shape = patch_shape

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
            'blocks': self.blocks,
            'patch_shape': self.patch_shape,
        })

    def build_model(self):
        """
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        def residual_block_down(x, filters, preactive=True, downsample=False):
            h = x

            if preactive:
                h = tf.keras.layers.ReLU()(x)

            h = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(h)
            h = tf.keras.layers.LeakyReLU(alpha=0.01)(h)
            h = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same'))(h)

            if preactive:
                x = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=1))(x)

            # if downsample:
            #     h = SpectralNormalization(
            #         tf.keras.layers.Conv2D(filters, kernel_size=2, strides=2, padding='same'))(h)
            #     x = SpectralNormalization(
            #         tf.keras.layers.Conv2D(filters, kernel_size=2, strides=2, padding='same'))(x)

            if downsample:
                h = tf.keras.layers.AveragePooling2D(pool_size=2, strides=downsample, padding='same')(h)
                x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=downsample, padding='same')(x)

            if not preactive:
                x = SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size=1))(x)

            return tf.keras.layers.Add()([h, x])

        image_inputs = tf.keras.layers.Input(shape=self.image_shape)
        block = ExtractPatches(patch_shape=self.patch_shape)(image_inputs)

        for i, filters in enumerate(self.blocks):
            downsample = (2, 2) if (i < len(self.blocks) - 1) else (1, 2)
            block = residual_block_down(block, filters, preactive=(i > 0), downsample=downsample)

            # if i == len(self.blocks) - 2:
            #     block = SelfAttention(spectral_norm=True)(block)

        block = tf.keras.layers.LeakyReLU(alpha=0.01)(block)

        # if self.patch_shape is None:
        #     block = MaskingPadding()([image_inputs, block])

        block = tf.keras.layers.Flatten()(block)
        outputs = SpectralNormalization(tf.keras.layers.Dense(units=1))(block)

        self.model = tf.keras.Model(inputs=image_inputs, outputs=outputs, name=self.name)


class BackboneModel(BaseModel):
    """
    A backbone model that extracts style patterns from images.

    This model is designed to capture the characteristics from input images,
        providing a basis for further style-related processing in generative tasks.
    """

    def __init__(self,
                 image_shape,
                 blocks,
                 name='backbone',
                 **kwargs):
        """
        Initialize the backbone model with specified parameters.

        Parameters
        ----------
        image_shape : list or tuple
            Shape of the input image.
        blocks : list or tuple
            Blocks of channels for the model's architecture.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name=name, **kwargs)

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
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        image_inputs = tf.keras.layers.Input(shape=self.image_shape)

        conv = tf.keras.layers.Conv2D(self.blocks[0], kernel_size=5, strides=2, padding='same')(image_inputs)
        blocks = list(self.blocks) + [self.blocks[-1] * 2]
        feats = []

        for i, filters in enumerate(self.blocks):
            dropout_rate = 0.1 if i <= (len(self.blocks) - 1) // 2 else 0.2

            block1 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv)
            block1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(block1)
            block1 = tf.keras.layers.BatchNormalization(renorm=True)(block1)
            block1 = tf.keras.layers.Dropout(rate=dropout_rate)(block1)

            block1 = tf.keras.layers.LeakyReLU(alpha=0.01)(block1)
            block1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(block1)
            block1 = tf.keras.layers.BatchNormalization(renorm=True)(block1)

            conv = tf.keras.layers.Add()([conv, block1])

            block2 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv)
            block2 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(block2)
            block2 = tf.keras.layers.BatchNormalization(renorm=True)(block2)
            block2 = tf.keras.layers.Dropout(rate=dropout_rate)(block2)

            block2 = tf.keras.layers.LeakyReLU(alpha=0.01)(block2)
            block2 = tf.keras.layers.Conv2D(blocks[i + 1], kernel_size=3, strides=1, padding='same')(block2)
            block2 = tf.keras.layers.BatchNormalization(renorm=True)(block2)

            shortcut = tf.keras.layers.Conv2D(blocks[i + 1],
                                              kernel_size=1,
                                              strides=1,
                                              padding='valid',
                                              use_bias=False)(conv)

            conv = tf.keras.layers.Add()([shortcut, block2])

            strides = (2, 2) if i < (len(self.blocks) - 1) // 2 else (1, 2)
            conv = tf.keras.layers.MaxPool2D(pool_size=3, strides=strides, padding='same')(conv)
            # conv = tf.keras.layers.Conv2D(blocks[i + 1], kernel_size=3, strides=strides, padding='same')(conv)

            if i >= len(self.blocks[:-3]):
                feats.append(conv)

        conv = tf.keras.layers.LeakyReLU(alpha=0.01)(conv)

        conv = tf.keras.layers.Conv2D(blocks[-1], kernel_size=3, strides=1, padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)
        conv = tf.keras.layers.LeakyReLU(alpha=0.01)(conv)

        outputs = tf.keras.layers.Reshape(target_shape=(conv.get_shape()[1], -1))(conv)
        # outputs = MaskingPadding()([image_inputs, outputs])

        self.features_output_shape = outputs.get_shape()[1:]
        self.model = tf.keras.Model(inputs=image_inputs, outputs=[outputs, feats], name=self.name)


class StyleEncoderModel(BaseModel):
    """
    An encoder model that encodes extracted style features from images into a representative style vector.

    This model is part of a generative architecture where style features are encoded into a latent
        representation, facilitating the generation of images with specific stylistic attributes.
    """

    def __init__(self,
                 features_shape,
                 latent_dim,
                 name='style_encoder',
                 **kwargs):
        """
        Initialize the style encoder model with specified parameters.

        Parameters
        ----------
        features_shape : list or tuple
            Shape of the input features.
        latent_dim : int
            Dimension of the latent space.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name=name, **kwargs)

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
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        feature_inputs = tf.keras.layers.Input(shape=self.features_shape)

        style = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name='expand')(feature_inputs)
        style = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(style)
        style = tf.keras.layers.Flatten()(style)

        style = tf.keras.layers.Dense(256)(style)
        style = tf.keras.layers.LeakyReLU(alpha=0.01)(style)

        style = tf.keras.layers.Dense(256)(style)
        style = tf.keras.layers.LeakyReLU(alpha=0.01)(style)

        mu = tf.keras.layers.Dense(self.latent_dim)(style)

        logvar = tf.keras.layers.Dense(self.latent_dim)(style)
        logvar = tf.keras.layers.LeakyReLU(alpha=0.01)(logvar)

        outputs = tf.keras.layers.Lambda(
            lambda x: x[0] + tf.exp(0.5 * x[1]) * tf.random.normal(tf.shape(x[1])), name='reparam')([mu, logvar])

        self.features_output_shape = outputs.get_shape()[1:]
        self.model = tf.keras.Model(inputs=feature_inputs, outputs=[outputs, mu, logvar], name=self.name)


class IdentificationModel(BaseModel):
    """
    A writer identification model that classifies handwriting images based on extracted style features.

    This model is designed to identify the writer of a given handwriting sample
        by analyzing the stylistic features of the handwriting.
    """

    def __init__(self,
                 features_shape,
                 writers_shape,
                 name='identification',
                 **kwargs):
        """
        Initialize the writer identification model with specified parameters.

        Parameters
        ----------
        features_shape : list or tuple
            Shape of the input features.
        writers_shape : int
            Number of writers to classify.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name=name, **kwargs)

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
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        feature_inputs = tf.keras.layers.Input(shape=self.features_shape)

        style = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name='expand')(feature_inputs)
        style = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(style)
        style = tf.keras.layers.Flatten()(style)

        style = tf.keras.layers.Dense(256)(style)
        style = tf.keras.layers.LeakyReLU(alpha=0.01)(style)

        outputs = tf.keras.layers.Dense(self.writers_shape[0])(style)

        self.model = tf.keras.Model(inputs=feature_inputs, outputs=outputs, name=self.name)


class RecognitionModel(BaseModel):
    """
    A recognition model that transcribes handwritten texts from images.

    This model is designed to extract textual information from images of handwriting,
        facilitating tasks like optical character recognition and handwriting analysis.
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 blocks,
                 name='recognition',
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
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name=name, **kwargs)

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
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        image_inputs = tf.keras.layers.Input(shape=self.image_shape)

        conv = tf.keras.layers.Conv2D(self.blocks[0], kernel_size=5, strides=2, padding='same')(image_inputs)
        blocks = list(self.blocks) + [self.blocks[-1] * 2]

        for i, filters in enumerate(self.blocks):
            dropout_rate = 0.1 if i <= (len(self.blocks) - 1) // 2 else 0.2

            block1 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv)
            block1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(block1)
            block1 = tf.keras.layers.BatchNormalization(renorm=True)(block1)
            block1 = tf.keras.layers.Dropout(rate=dropout_rate)(block1)

            block1 = tf.keras.layers.LeakyReLU(alpha=0.01)(block1)
            block1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(block1)
            block1 = tf.keras.layers.BatchNormalization(renorm=True)(block1)

            conv = tf.keras.layers.Add()([conv, block1])

            block2 = tf.keras.layers.LeakyReLU(alpha=0.01)(conv)
            block2 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(block2)
            block2 = tf.keras.layers.BatchNormalization(renorm=True)(block2)
            block2 = tf.keras.layers.Dropout(rate=dropout_rate)(block2)

            block2 = tf.keras.layers.LeakyReLU(alpha=0.01)(block2)
            block2 = tf.keras.layers.Conv2D(blocks[i + 1], kernel_size=3, strides=1, padding='same')(block2)
            block2 = tf.keras.layers.BatchNormalization(renorm=True)(block2)

            shortcut = tf.keras.layers.Conv2D(blocks[i + 1],
                                              kernel_size=1,
                                              strides=1,
                                              padding='valid',
                                              use_bias=False)(conv)

            conv = tf.keras.layers.Add()([shortcut, block2])

            strides = (2, 2) if i < (len(self.blocks) - 1) // 2 else (1, 2)
            conv = tf.keras.layers.Conv2D(blocks[i + 1], kernel_size=3, strides=strides, padding='same')(conv)

        conv = tf.keras.layers.LeakyReLU(alpha=0.01)(conv)

        conv = tf.keras.layers.Conv2D(blocks[-1], kernel_size=3, strides=(1, 2), padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization(renorm=True)(conv)
        conv = tf.keras.layers.LeakyReLU(alpha=0.01)(conv)

        conv = tf.keras.layers.Reshape(target_shape=(conv.get_shape()[1], -1))(conv)
        # conv = MaskingPadding()([image_inputs, conv])

        blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(conv)
        blstm = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(blstm)

        outputs = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-2), name='expand')(blstm)

        self.model = tf.keras.Model(inputs=image_inputs, outputs=outputs, name=self.name)
