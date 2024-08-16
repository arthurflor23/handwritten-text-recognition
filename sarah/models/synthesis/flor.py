import tensorflow as tf

from sarah.models.components.base import BaseModel
from sarah.models.components.base import BaseSynthesisModel
from sarah.models.components.layers import ConditionalBatchNormalization
from sarah.models.components.layers import ExtractPatches
from sarah.models.components.layers import GatedConv2D
from sarah.models.components.layers import Reparameterization
from sarah.models.components.layers import SelfAttention
from sarah.models.components.metrics import MetricsTracker
from sarah.models.recognition.flor_v2 import RecognitionModel


class SynthesisModel(BaseSynthesisModel):
    """
    This model integrates several submodels including a handwriting recognition,
        writer identifier, style encoder, generator, and discriminator.

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

    Image-to-Image Translation with Conditional Adversarial Networks
        https://arxiv.org/abs/1611.07004

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
            learning_rate = 1e-4

        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.95, epsilon=1e-8)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.95, epsilon=1e-8)

        self.r_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.95, epsilon=1e-8)
        self.w_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.95, epsilon=1e-8)

    def build_model(self):
        """
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        latent_dim = 128
        embedding_dim = 32
        patch_shape = [32, 32, 1]
        discriminator_blocks = [64, 128, 256, 256]
        generator_blocks = [256, 128, 64, 64]

        self.recognition = RecognitionModel(name='recognition',
                                            image_shape=self.image_shape,
                                            lexical_shape=self.lexical_shape)

        self.style_backbone = BackboneModel(name='style_backbone',
                                            image_shape=self.image_shape,
                                            backbone=self.recognition.encoder)

        self.identification = IdentificationModel(name='identification',
                                                  features_shape=self.style_backbone._output_shape,
                                                  writers_shape=self.writers_shape)

        self.style_encoder = StyleEncoderModel(name='style_encoder',
                                               features_shape=self.style_backbone._output_shape,
                                               latent_dim=latent_dim)

        self.generator = GeneratorModel(name='generator',
                                        latent_dim=latent_dim,
                                        embedding_dim=embedding_dim,
                                        image_shape=self.image_shape,
                                        lexical_shape=self.lexical_shape,
                                        blocks=generator_blocks)

        self.discriminator = DiscriminatorModel(name='discriminator',
                                                image_shape=self.image_shape,
                                                blocks=discriminator_blocks)

        self.patch_discriminator = DiscriminatorModel(name='patch_discriminator',
                                                      image_shape=self.image_shape,
                                                      blocks=discriminator_blocks,
                                                      patch_shape=patch_shape)

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

        self.discriminator.trainable = True
        self.patch_discriminator.trainable = True
        self.style_backbone.trainable = True
        self.identification.trainable = True
        self.recognition.trainable = True
        self.style_encoder.trainable = False
        self.generator.trainable = False

        for _ in range(self.discriminator_steps):
            random_latent_shape = (tf.shape(image_data)[0], self.style_encoder.latent_dim)
            random_latent_data = tf.stop_gradient(tf.random.normal(shape=random_latent_shape))

            real_features_data, _ = self.style_backbone(image_data, training=True)
            real_latent_data, _, _ = self.style_encoder(real_features_data, training=True)

            real_s_real_t_images = self.generator([real_latent_data, text_data], training=True)
            real_s_fake_t_images = self.generator([real_latent_data, aug_text_data], training=True)
            fake_s_fake_t_images = self.generator([random_latent_data, aug_text_data], training=True)

            fake_images = tf.concat([real_s_real_t_images, real_s_fake_t_images, fake_s_fake_t_images], axis=0)

            # patch and discriminator
            with tf.GradientTape() as tape:
                # fake images
                fake_adv = self.discriminator(fake_images, training=True)
                fake_adv_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_adv))

                # fake_patch_adv = self.patch_discriminator(fake_images, training=True)
                # fake_patch_adv_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_patch_adv))

                # real images
                read_image_adv = self.discriminator(image_data, training=True)
                real_aug_image_adv = self.discriminator(aug_image_data, training=True)

                real_adv = tf.concat([read_image_adv, real_aug_image_adv], axis=0)
                real_adv_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_adv))

                # read_image_patch_adv = self.patch_discriminator(image_data, training=True)
                # real_aug_image_patch_adv = self.patch_discriminator(aug_image_data, training=True)

                # real_patch_adv = tf.concat([read_image_patch_adv, real_aug_image_patch_adv], axis=0)
                # real_patch_adv_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_patch_adv))

                # discriminator loss
                d_adv_loss = fake_adv_loss + real_adv_loss  # + fake_patch_adv_loss + real_patch_adv_loss

            d_gradients = tape.gradient(d_adv_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

            # d_gradients = tape.gradient(d_adv_loss, self.discriminator.trainable_variables +
            #                             self.patch_discriminator.trainable_weights)

            # self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables +
            #                                      self.patch_discriminator.trainable_weights))

            # handwriting recognition
            with tf.GradientTape() as tape:
                ctc_logits = self.recognition(aug_image_data, training=True)
                d_ctc_loss = self.ctc_loss(text_data, ctc_logits)

            r_gradients = tape.gradient(d_ctc_loss, self.recognition.trainable_weights)
            self.r_optimizer.apply_gradients(zip(r_gradients, self.recognition.trainable_weights))

            # writer identifier
            with tf.GradientTape() as tape:
                wid_features_data, _ = self.style_backbone(aug_image_data, training=True)
                wid_logits = self.identification(wid_features_data, training=True)
                d_wid_loss = self.cls_loss(writer_data, wid_logits)

            w_gradients = tape.gradient(d_wid_loss, self.style_backbone.trainable_weights +
                                        self.identification.trainable_weights)

            self.w_optimizer.apply_gradients(zip(w_gradients, self.style_backbone.trainable_weights +
                                                 self.identification.trainable_weights))

        self.metrics_tracker.update({
            'd_adv_loss': d_adv_loss,
            'd_ctc_loss': d_ctc_loss,
            'd_wid_loss': d_wid_loss,
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

        random_latent_shape = (tf.shape(image_data)[0], self.style_encoder.latent_dim)
        random_latent_data = tf.stop_gradient(tf.random.normal(shape=random_latent_shape))

        self.discriminator.trainable = False
        self.patch_discriminator.trainable = False
        self.style_backbone.trainable = False
        self.identification.trainable = False
        self.recognition.trainable = False
        self.style_encoder.trainable = True
        self.generator.trainable = True

        with tf.GradientTape() as tape:
            real_features_data, real_feats = self.style_backbone(image_data, training=True)
            real_latent_data, mu, logvar = self.style_encoder(real_features_data, training=True)

            real_s_real_t_images = self.generator([real_latent_data, text_data], training=True)
            real_s_fake_t_images = self.generator([real_latent_data, aug_text_data], training=True)
            fake_s_fake_t_images = self.generator([random_latent_data, aug_text_data], training=True)

            fake_images = tf.concat([real_s_real_t_images, real_s_fake_t_images, fake_s_fake_t_images], axis=0)

            # patch and discriminator (adversarial)
            fake_adv_adv = self.discriminator(fake_images, training=True)
            fake_adv_loss = -tf.reduce_mean(fake_adv_adv)

            # fake_patch_adv_adv = self.patch_discriminator(fake_images, training=True)
            # fake_patch_adv_loss = -tf.reduce_mean(fake_patch_adv_adv)

            g_adv_loss = fake_adv_loss  # + fake_patch_adv_loss

            # handwriting recognition
            real_s_real_t_ctc = self.recognition(real_s_real_t_images, training=True)
            real_s_fake_t_ctc = self.recognition(real_s_fake_t_images, training=True)
            fake_s_fake_t_ctc = self.recognition(fake_s_fake_t_images, training=True)

            real_texts = tf.concat([text_data, aug_text_data, aug_text_data], axis=0)
            fake_texts = tf.concat([real_s_real_t_ctc, real_s_fake_t_ctc, fake_s_fake_t_ctc], axis=0)

            g_ctc_loss = self.ctc_loss(real_texts, fake_texts)

            # content reconstruction
            g_rec_loss = self.bva_loss(image_data, (real_s_real_t_images, real_latent_data, mu, logvar))

            # style reconstruction
            fake_features_data, _ = self.style_backbone(fake_s_fake_t_images, training=True)
            fake_latent_data, _, _ = self.style_encoder(fake_features_data, training=True)
            g_res_loss = tf.reduce_mean(tf.math.abs(fake_latent_data - random_latent_data))

            # writer identifier
            real_t_features_data, real_t_feats = self.style_backbone(real_s_real_t_images, training=True)
            real_t_wid_logits = self.identification(real_t_features_data, training=True)

            fake_t_features_data, fake_t_feats = self.style_backbone(real_s_fake_t_images, training=True)
            fake_t_wid_logits = self.identification(fake_t_features_data, training=True)

            real_fake_t_wid_logits = tf.concat([real_t_wid_logits, fake_t_wid_logits], axis=0)
            g_wid_loss = self.cls_loss(tf.repeat(writer_data, repeats=2, axis=0), real_fake_t_wid_logits)

            # contextual
            g_ctx_loss = tf.constant(0.0)
            g_ctx_loss += self.ctx_loss(real_feats, real_t_feats)
            g_ctx_loss += self.ctx_loss(real_feats, fake_t_feats)

            # generator loss
            g_loss = g_adv_loss + g_ctc_loss + g_rec_loss + g_res_loss + g_wid_loss + g_ctx_loss

        g_gradients = tape.gradient(g_loss, self.style_encoder.trainable_weights +
                                    self.generator.trainable_weights)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.style_encoder.trainable_weights +
                                             self.generator.trainable_weights))

        self.metrics_tracker.update({
            'g_adv_loss': g_adv_loss,
            'g_ctc_loss': g_ctc_loss,
            'g_ctx_loss': g_ctx_loss,
            'g_rec_loss': g_rec_loss,
            'g_res_loss': g_res_loss,
            'g_wid_loss': g_wid_loss,
            'loss': g_loss,
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


class BackboneModel(BaseModel):
    """
    A backbone model that extracts style patterns from images.

    This model is designed to capture the characteristics from input images,
        providing a basis for further style-related processing in generative tasks.
    """

    def __init__(self,
                 image_shape,
                 backbone,
                 name='style_backbone',
                 **kwargs):
        """
        Initialize the backbone model with specified parameters.

        Parameters
        ----------
        image_shape : list or tuple
            Shape of the input image.
        backbone : tf.keras.Model
            Backbone model for feature extraction.
        name : str, optional
            A name for the instance.
        **kwargs : dict
            Additional keyword arguments.
        """

        super().__init__(name=name, **kwargs)

        self.image_shape = image_shape
        self.backbone = backbone

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
        })

    def build_model(self):
        """
        Builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        style = tf.keras.layers.Reshape(target_shape=(-1, self.backbone.output.shape[-1]))(self.backbone.output)
        style = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name='expand')(style)

        style = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(style)
        style = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(style)

        outputs = tf.keras.layers.Flatten()(style)

        self._input_shape = self.backbone.output.shape[1:]
        self._output_shape = outputs.shape[1:]

        self.model = tf.keras.Model(name=self.name, inputs=self.backbone.input, outputs=[outputs, self.backbone.output])


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

        style = tf.keras.layers.Dense(units=256)(feature_inputs)
        style = tf.keras.layers.LeakyReLU(negative_slope=0.2)(style)

        outputs = tf.keras.layers.Dense(units=self.writers_shape[0])(style)

        self.model = tf.keras.Model(name=self.name, inputs=feature_inputs, outputs=outputs)


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

        style = tf.keras.layers.Dense(units=256)(feature_inputs)
        style = tf.keras.layers.LeakyReLU(negative_slope=0.2)(style)

        mu = tf.keras.layers.Dense(units=self.latent_dim)(style)
        logvar = tf.keras.layers.Dense(units=self.latent_dim)(style)

        outputs = Reparameterization()([mu, logvar])

        self.model = tf.keras.Model(name=self.name, inputs=feature_inputs, outputs=[outputs, mu, logvar])


class GeneratorModel(BaseModel):
    """
    A generator model that combines latent and vocabulary data for generative tasks.

    This model synthesizes images based on the combination of latent space vectors and
        lexical information, suitable for tasks like image generation from textual descriptions.
    """

    def __init__(self,
                 latent_dim,
                 embedding_dim,
                 image_shape,
                 lexical_shape,
                 blocks,
                 name='generator',
                 **kwargs):
        """
        Initialize the generator model with specified parameters.

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space.
        embedding_dim : int
            Dimension of the embedding space.
        image_shape : list or tuple
            Shape of the output image.
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

        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
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
            'latent_dim': self.latent_dim,
            'embedding_dim': self.embedding_dim,
            'image_shape': self.image_shape,
            'lexical_shape': self.lexical_shape,
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
            h = ConditionalBatchNormalization(spectral=True)([x, y])
            h = tf.keras.layers.LeakyReLU(negative_slope=0.2)(h)

            if upsample:
                h = tf.keras.layers.UpSampling2D(size=upsample)(h)
                x = tf.keras.layers.UpSampling2D(size=upsample)(x)

                # h = tf.keras.layers.SpectralNormalization(
                #     tf.keras.layers.Conv2DTranspose(filters=filters,
                #                                     kernel_size=2,
                #                                     strides=upsample,
                #                                     padding='same',
                #                                     kernel_initializer='orthogonal'))(h)

                # x = tf.keras.layers.SpectralNormalization(
                #     tf.keras.layers.Conv2DTranspose(filters=filters,
                #                                     kernel_size=2,
                #                                     strides=upsample,
                #                                     padding='same',
                #                                     kernel_initializer='orthogonal'))(x)

            h = tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=3,
                                       strides=1,
                                       padding='same',
                                       kernel_initializer='orthogonal'))(h)

            h = ConditionalBatchNormalization(spectral=True)([h, y])
            h = tf.keras.layers.LeakyReLU(negative_slope=0.2)(h)

            h = tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=3,
                                       strides=1,
                                       padding='same',
                                       kernel_initializer='orthogonal'))(h)

            x = tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=1,
                                       strides=1,
                                       padding='valid',
                                       kernel_initializer='orthogonal'))(x)

            return tf.keras.layers.Add()([h, x])

        latent_input = tf.keras.layers.Input(shape=(self.latent_dim,))

        text_input = tf.keras.layers.Input(shape=self.lexical_shape[:-1])
        text_flattened = tf.keras.layers.Flatten()(text_input)

        text_embedding = tf.keras.layers.Embedding(input_dim=self.lexical_shape[-1],
                                                   output_dim=self.embedding_dim,
                                                   embeddings_initializer='orthogonal')(text_flattened)

        latent_dense = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Dense(units=self.latent_dim * len(self.blocks),
                                  kernel_initializer='orthogonal'))(latent_input)

        latent_chunks = tf.keras.layers.Lambda(function=lambda x, y: tf.split(x, num_or_size_splits=y, axis=1),
                                               arguments={'y': len(self.blocks)},
                                               name='chunks')(latent_dense)

        latent_expanded = tf.keras.layers.Lambda(function=lambda x: tf.expand_dims(x, axis=1),
                                                 name='expand')(latent_input)

        latent_tiled = tf.keras.layers.Lambda(function=lambda x, y: tf.tile(x, y),
                                              arguments={'y': [1, text_embedding.shape[1], 1]},
                                              name='tile')(latent_expanded)

        latent_text = tf.keras.layers.Concatenate(axis=-1)([latent_tiled, text_embedding])

        latent_text = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Dense(units=4 * 4 * 2 * self.blocks[0],
                                  kernel_initializer='orthogonal'))(latent_text)

        block = tf.keras.layers.Reshape(target_shape=(latent_text.shape[1] * 4, 4, -1))(latent_text)
        # block = tf.keras.layers.Reshape(target_shape=(latent_text.shape[1], 4 * 4, -1))(latent_text)

        for i, filters in enumerate(self.blocks):
            # if i == len(self.blocks) - 1:
            #     block = GatedConv2D(mode='residual', kernel_initializer='orthogonal')(block)

            strides = (2 if block.shape[1] < self.image_shape[0] else 1,
                       2 if block.shape[2] < self.image_shape[1] else 1)

            upsample = strides if 2 in strides else None
            block = residual_block_up(block, latent_chunks[i], filters, upsample=upsample)

            # if i == 0:
            #     block = SelfAttention(pooling=True, spectral=True, kernel_initializer='orthogonal')(block)

        outputs = tf.keras.layers.BatchNormalization()(block)
        outputs = tf.keras.layers.LeakyReLU(negative_slope=0.2)(outputs)

        outputs = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=1,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   kernel_initializer='orthogonal'))(outputs)

        outputs = tf.keras.layers.Activation(activation='tanh')(outputs)

        self.model = tf.keras.Model(name=self.name, inputs=[latent_input, text_input], outputs=outputs)


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

        def residual_block_down(x, filters, preactive=True, downsample=None):
            h = x

            if preactive:
                h = tf.keras.layers.LeakyReLU(negative_slope=0.2)(h)

            h = tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=3,
                                       padding='same',
                                       kernel_initializer='orthogonal'))(h)
            h = tf.keras.layers.LeakyReLU(negative_slope=0.2)(h)

            h = tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=3,
                                       padding='same',
                                       kernel_initializer='orthogonal'))(h)

            if preactive:
                x = tf.keras.layers.SpectralNormalization(
                    tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=1,
                                           padding='valid',
                                           kernel_initializer='orthogonal'))(x)

            if downsample:
                h = tf.keras.layers.AveragePooling2D(pool_size=2, strides=downsample, padding='same')(h)
                x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=downsample, padding='same')(x)

                # h = tf.keras.layers.SpectralNormalization(
                #     tf.keras.layers.Conv2DTranspose(filters=filters,
                #                                     kernel_size=2,
                #                                     strides=downsample,
                #                                     padding='same',
                #                                     kernel_initializer='orthogonal'))(h)

                # x = tf.keras.layers.SpectralNormalization(
                #     tf.keras.layers.Conv2DTranspose(filters=filters,
                #                                     kernel_size=2,
                #                                     strides=downsample,
                #                                     padding='same',
                #                                     kernel_initializer='orthogonal'))(x)

            if not preactive:
                x = tf.keras.layers.SpectralNormalization(
                    tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=1,
                                           padding='valid',
                                           kernel_initializer='orthogonal'))(x)

            return tf.keras.layers.Add()([h, x])

        image_inputs = tf.keras.layers.Input(shape=self.image_shape)
        block = ExtractPatches(patch_shape=self.patch_shape)(image_inputs)

        for i, filters in enumerate(self.blocks):
            # if i == len(self.blocks) - 1:
            #     block = SelfAttention(pooling=True, spectral=True, kernel_initializer='orthogonal')(block)

            strides = (2 if block.shape[1] > 4 else 1,
                       2 if block.shape[2] > 4 else 1)

            downsample = strides if 2 in strides else None
            block = residual_block_down(block, filters, preactive=(i > 0), downsample=downsample)

            # if i == 0:
            #     block = GatedConv2D(mode='residual', kernel_initializer='orthogonal')(block)

        outputs = tf.keras.layers.GlobalAveragePooling2D()(block)

        outputs = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Dense(units=1, kernel_initializer='orthogonal'))(outputs)

        self.model = tf.keras.Model(name=self.name, inputs=image_inputs, outputs=outputs)
