import tensorflow as tf

from sarah.models.components.base import BaseModel
from sarah.models.components.base import BaseSynthesisModel
from sarah.models.components.layers import AdaptiveInstanceNormalization
from sarah.models.components.layers import ConditionalBatchNormalization
from sarah.models.components.layers import ExtractPatches
from sarah.models.components.layers import GatedConv2DResidual
from sarah.models.components.layers import PositionEmbedding
from sarah.models.components.layers import Reparameterization
from sarah.models.components.metrics import MetricsTracker
from sarah.models.recognition.flor_v2 import RecognitionModel as HTRModel


class SynthesisModel(BaseSynthesisModel):
    """
    Integrates multiple submodels for handwriting recognition, writer identification,
        style encoding, generation, and discrimination.

    References
    ----------
    Adversarial Generation of Handwritten Text Images Conditioned on Sequences
        https://arxiv.org/abs/1903.00277

    Conditional Generative Adversarial Nets
        https://arxiv.org/abs/1411.1784

    HiGAN+: Handwriting Imitation GAN with Disentangled Representations
        https://dl.acm.org/doi/10.1145/3550070

    Image-to-Image Translation with Conditional Adversarial Networks
        https://arxiv.org/abs/1611.07004

    Large Scale GAN Training for High Fidelity Natural Image Synthesis
        https://arxiv.org/abs/1809.11096v2

    ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation
        https://arxiv.org/abs/2003.10557
    """

    def compile(self, learning_rate=None):
        """
        Compiles the model.

        Parameters
        ----------
        learning_rate : float, optional
            Optimizer learning rate.
        """

        super().compile(run_eagerly=False)

        if learning_rate is None:
            learning_rate = 1e-4

        self.r_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.95)
        self.w_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.95)

        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.95)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.95)

    def build_model(self):
        """
        Builds the model architecture.
        """

        text_dim = 32
        latent_dim = 128
        patch_shape = [32, 32]

        generator_blocks = [256, 128, 64, 32]
        discriminator_blocks = [32, 64, 128, 256]

        self.style_backbone = BackboneModel(name='style_backbone',
                                            image_shape=self.image_shape,
                                            model=HTRModel(image_shape=self.image_shape,
                                                           lexical_shape=self.lexical_shape).encoder)

        self.recognition = RecognitionModel(name='recognition',
                                            lexical_shape=self.lexical_shape,
                                            model=HTRModel(image_shape=self.image_shape,
                                                           lexical_shape=self.lexical_shape).recognition)

        self.identification = IdentificationModel(name='identification',
                                                  features_shape=self.style_backbone.model.output[0].shape[1:],
                                                  writers_shape=self.writers_shape)

        self.style_encoder = StyleEncoderModel(name='style_encoder',
                                               features_shape=self.style_backbone.model.output[0].shape[1:],
                                               mask_shape=self.image_shape,
                                               latent_dim=latent_dim)

        self.generator = GeneratorModel(name='generator',
                                        image_shape=self.image_shape,
                                        lexical_shape=self.lexical_shape,
                                        text_dim=text_dim,
                                        latent_dim=latent_dim,
                                        blocks=generator_blocks)

        self.discriminator = DiscriminatorModel(name='discriminator',
                                                image_shape=self.image_shape,
                                                blocks=discriminator_blocks,
                                                lexical_shape=self.lexical_shape)

        self.patch_discriminator = DiscriminatorModel(name='patch_discriminator',
                                                      image_shape=self.image_shape,
                                                      blocks=discriminator_blocks,
                                                      patch_shape=patch_shape)

        self.metrics_tracker = MetricsTracker()

    def discriminator_step(self, input_data):
        """
        Updates the discriminator.

        Parameters
        ----------
        input_data : list or tuple
            Batch of (x_data, y_data).
        """

        x_data, y_data = input_data

        (aug_image_data, aug_text_data, _, _) = x_data
        (image_data, text_data, writer_data, mask_data) = y_data

        self.discriminator.trainable = True
        self.patch_discriminator.trainable = True
        self.recognition.trainable = True
        self.style_backbone.trainable = True
        self.identification.trainable = True
        self.style_encoder.trainable = False
        self.generator.trainable = False

        for _ in range(self.discriminator_steps):
            random_latent_shape = (tf.shape(image_data)[0], self.style_encoder.latent_dim)
            random_latent_data = tf.stop_gradient(tf.random.normal(shape=random_latent_shape))

            real_features_data, _ = self.style_backbone(image_data, training=True)
            real_latent_data, _, _ = self.style_encoder([real_features_data, mask_data], training=True)

            real_real_images = self.generator([text_data, real_latent_data], training=True)
            real_fake_images = self.generator([aug_text_data, real_latent_data], training=True)
            fake_fake_images = self.generator([aug_text_data, random_latent_data], training=True)

            real_images = tf.concat([image_data, aug_image_data], axis=0)
            fake_images = tf.concat([real_real_images, real_fake_images, fake_fake_images], axis=0)

            # discriminator and patch discriminator
            with tf.GradientTape() as d_tape:
                # fake images
                fake_adv = self.discriminator(fake_images, training=True)
                fake_adv_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_adv))

                # fake_patch_adv = self.patch_discriminator(fake_images, training=True)
                # fake_patch_adv_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_patch_adv))

                # real images
                real_adv = self.discriminator(real_images, training=True)
                real_adv_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_adv))

                # real_patch_adv = self.patch_discriminator(real_images, training=True)
                # real_patch_adv_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_patch_adv))

                # discriminator loss
                d_adv_loss = fake_adv_loss + real_adv_loss
                # d_adv_loss = fake_adv_loss + real_adv_loss + fake_patch_adv_loss + real_patch_adv_loss

            d_gradients = d_tape.gradient(d_adv_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

            # d_gradients = tape.gradient(d_adv_loss,
            #                             self.discriminator.trainable_variables +
            #                             self.patch_discriminator.trainable_weights)

            # self.d_optimizer.apply_gradients(zip(d_gradients,
            #                                      self.discriminator.trainable_variables +
            #                                      self.patch_discriminator.trainable_weights))

            # handwriting recognition
            with tf.GradientTape() as r_tape:
                ctc_logits = self.recognition(image_data, training=True)
                d_ctc_loss = self.ctc_loss(text_data, ctc_logits)

            r_gradients = r_tape.gradient(d_ctc_loss, self.recognition.trainable_weights)
            self.r_optimizer.apply_gradients(zip(r_gradients, self.recognition.trainable_weights))

            # writer identification
            with tf.GradientTape() as w_tape:
                wid_features_data, _ = self.style_backbone(image_data, training=True)
                wid_logits = self.identification(wid_features_data, training=True)
                d_wid_loss = self.cls_loss(writer_data, wid_logits)

            w_gradients = w_tape.gradient(d_wid_loss,
                                          self.style_backbone.trainable_weights +
                                          self.identification.trainable_weights)

            self.w_optimizer.apply_gradients(zip(w_gradients,
                                                 self.style_backbone.trainable_weights +
                                                 self.identification.trainable_weights))

        self.metrics_tracker.update({
            'd_adv_loss': d_adv_loss,
            'd_ctc_loss': d_ctc_loss,
            'd_wid_loss': d_wid_loss,
        })

    def generator_step(self, input_data):
        """
        Updates the generator.

        Parameters
        ----------
        input_data : list or tuple
            Batch of (x_data, y_data).
        """

        x_data, y_data = input_data

        (_, aug_text_data, _, _) = x_data
        (image_data, text_data, writer_data, mask_data) = y_data

        random_latent_shape = (tf.shape(image_data)[0], self.style_encoder.latent_dim)
        random_latent_data = tf.stop_gradient(tf.random.normal(shape=random_latent_shape))

        self.discriminator.trainable = False
        self.patch_discriminator.trainable = False
        self.recognition.trainable = False
        self.style_backbone.trainable = False
        self.identification.trainable = False
        self.style_encoder.trainable = True
        self.generator.trainable = True

        with tf.GradientTape() as g_tape:
            real_features, real_feats = self.style_backbone(image_data, training=True)
            real_latent_data, mu, logvar = self.style_encoder([real_features, mask_data], training=True)

            real_real_images = self.generator([text_data, real_latent_data], training=True)
            real_fake_images = self.generator([aug_text_data, real_latent_data], training=True)
            fake_fake_images = self.generator([aug_text_data, random_latent_data], training=True)

            fake_images = tf.concat([real_real_images, real_fake_images, fake_fake_images], axis=0)
            real_texts = tf.concat([text_data, aug_text_data, aug_text_data], axis=0)

            # discriminator and patch discriminator
            fake_adv_adv = self.discriminator(fake_images, training=True)
            fake_adv_loss = -tf.reduce_mean(fake_adv_adv)

            # fake_patch_adv_adv = self.patch_discriminator(fake_images, training=True)
            # fake_patch_adv_loss = -tf.reduce_mean(fake_patch_adv_adv)

            g_adv_loss = fake_adv_loss
            # g_adv_loss = fake_adv_loss + fake_patch_adv_loss

            # handwriting recognition
            fake_texts = self.recognition(fake_images, training=True)
            g_ctc_loss = self.ctc_loss(real_texts, fake_texts)

            # content reconstruction
            g_rec_loss = self.bva_loss(image_data, (real_real_images, real_latent_data, mu, logvar))

            # style reconstruction
            fake_fake_features, _ = self.style_backbone(fake_fake_images, training=True)
            fake_latent_data, _, _ = self.style_encoder([fake_fake_features, mask_data], training=True)

            g_res_loss = tf.reduce_mean(tf.math.abs(fake_latent_data - random_latent_data))

            # writer identifier
            real_latent_images = tf.concat([real_real_images, real_fake_images], axis=0)

            real_latent_features, real_latent_feats = self.style_backbone(real_latent_images, training=True)
            real_latent_wid_logits = self.identification(real_latent_features, training=True)

            g_wid_loss = self.cls_loss(tf.repeat(writer_data, repeats=2, axis=0), real_latent_wid_logits)

            # contextual
            g_ctx_loss = tf.constant(0.0)

            for real_feat, real_latent_feat in zip(real_feats, real_latent_feats):
                feats = tf.split(real_latent_feat, num_or_size_splits=2, axis=0)

                g_ctx_loss += self.ctx_loss(real_feat, feats[0])
                g_ctx_loss += self.ctx_loss(real_feat, feats[1])

            # generator loss
            g_loss = g_adv_loss + g_ctc_loss + g_ctx_loss + g_rec_loss + g_res_loss + g_wid_loss

        g_gradients = g_tape.gradient(g_loss,
                                      self.style_encoder.trainable_weights +
                                      self.generator.trainable_weights)

        self.g_optimizer.apply_gradients(zip(g_gradients,
                                             self.style_encoder.trainable_weights +
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
        Executes a training step.

        Parameters
        ----------
        input_data : list or tuple
            Batch of (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        self.discriminator_step(input_data)

        tf.cond(pred=tf.math.equal(tf.math.mod(self.global_steps, self.generator_steps), 0),
                true_fn=lambda: self.generator_step(input_data),
                false_fn=lambda: None)

        self.global_steps.assign_add(delta=1)

        # kid metric
        _, (image_data, text_data, _, mask_data) = input_data

        features_data, _ = self.style_backbone(image_data, training=False)
        latent_data, _, _ = self.style_encoder([features_data, mask_data], training=False)
        generated_images = self.generator([text_data, latent_data], training=False)

        self.kid.update_state(image_data, generated_images)
        self.metrics_tracker.update({self.kid.name: self.kid.result()})

        return self.metrics_tracker.result()


class BackboneModel(BaseModel):
    """
    Extracts style patterns from input images for generative tasks.
    """

    def __init__(self,
                 image_shape,
                 model,
                 name='style_backbone',
                 **kwargs):
        """
        Initialize the backbone model.

        Parameters
        ----------
        image_shape : list or tuple
            Input image shape.
        model : tf.keras.Model
            Model for feature extraction.
        name : str, optional
            Model instance name.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.image_shape = image_shape
        self.model = model

        self.build_model()

        if hasattr(self, 'model'):
            self.summary = self.model.summary
            self.call = self.model.call

    def get_config(self):
        """
        Return the configuration of the model.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'image_shape': self.image_shape,
        })

    def build_model(self):
        """
        Builds the model architecture.
        """

        feat_layers = ['GatedConv2DResidual', 'SelfAttention']
        feats = [x.output for x in self.model.layers if x.__class__.__name__ in feat_layers]

        self.model = tf.keras.Model(name=self.name,
                                    inputs=self.model.input,
                                    outputs=[self.model.output, feats[-5:]])


class RecognitionModel(BaseModel):
    """
    Model for recognizing and decoding text sequences.
    """

    def __init__(self,
                 lexical_shape,
                 model,
                 name='recognition',
                 **kwargs):
        """
        Initializes the recognition model.

        Parameters
        ----------
        lexical_shape : list or tuple
            Shape of text sequences and vocabulary.
        model : tf.keras.Model
            Model for feature decoding.
        name : str, optional
            Model instance name.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.lexical_shape = lexical_shape
        self.model = model

        self.build_model()

        if hasattr(self, 'model'):
            self.summary = self.model.summary
            self.call = self.model.call

    def get_config(self):
        """
        Return the configuration of the model.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'lexical_shape': self.lexical_shape,
        })

    def build_model(self):
        """
        Builds the model architecture.
        """

        self.model = tf.keras.Model(name=self.name,
                                    inputs=self.model.input,
                                    outputs=self.model.output)


class IdentificationModel(BaseModel):
    """
    Classifies handwriting images to identify the writer based on style features.
    """

    def __init__(self,
                 features_shape,
                 writers_shape,
                 name='identification',
                 **kwargs):
        """
        Initializes the writer identification model.

        Parameters
        ----------
        features_shape : list or tuple
            Input feature shape.
        writers_shape : int
            Number of writers to classify.
        name : str, optional
            Model instance name.
        **kwargs : dict
            Additional arguments.
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
        Return the configuration of the model.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'features_shape': self.features_shape,
            'writers_shape': self.writers_shape,
        })

    def build_model(self):
        """
        Builds the model architecture.
        """

        feature_inputs = tf.keras.layers.Input(shape=self.features_shape)

        style = tf.keras.layers.GlobalAveragePooling2D()(feature_inputs)

        style = tf.keras.layers.Dense(units=256)(style)
        style = tf.keras.layers.Activation(activation='swish')(style)

        style = tf.keras.layers.Dense(units=256)(style)
        style = tf.keras.layers.Activation(activation='swish')(style)

        outputs = tf.keras.layers.Dense(units=self.writers_shape[0])(style)

        self.model = tf.keras.Model(name=self.name,
                                    inputs=feature_inputs,
                                    outputs=outputs)


class StyleEncoderModel(BaseModel):
    """
    Encodes style features from images into a latent style vector.
    """

    def __init__(self,
                 features_shape,
                 mask_shape,
                 latent_dim,
                 name='style_encoder',
                 **kwargs):
        """
        Initializes the style encoder model.

        Parameters
        ----------
        features_shape : list or tuple
            Input feature shape.
        mask_shape : list or tuple
            Input mask shape.
        latent_dim : int
            Dimension of the style latent space.
        name : str, optional
            Model instance name.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.features_shape = features_shape
        self.mask_shape = mask_shape
        self.latent_dim = latent_dim

        self.build_model()

        if hasattr(self, 'model'):
            self.summary = self.model.summary
            self.call = self.model.call

    def get_config(self):
        """
        Return the configuration of the model.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'features_shape': self.features_shape,
            'mask_shape': self.mask_shape,
            'latent_dim': self.latent_dim,
        })

    def build_model(self):
        """
        Builds the model architecture.
        """

        feature_inputs = tf.keras.layers.Input(shape=self.features_shape)

        style = tf.keras.layers.GlobalAveragePooling2D()(feature_inputs)

        mask_input = tf.keras.layers.Input(shape=self.mask_shape)
        # mask = tf.keras.layers.Identity()(mask_input)

        # for _ in range(6):
        #     mask = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(mask)
        #     mask = tf.keras.layers.Activation(activation='swish')(mask)

        #     if 1 in mask.shape[1:-1]:
        #         break

        # mask = tf.keras.layers.Flatten()(mask)
        # mask = tf.keras.layers.Dense(units=feature_inputs.shape[-1])(mask)

        # latent = tf.keras.layers.Concatenate(axis=-1)([style, mask])
        latent = tf.keras.layers.Identity()(style)

        latent = tf.keras.layers.Dense(units=256)(latent)
        latent = tf.keras.layers.Activation(activation='swish')(latent)

        latent = tf.keras.layers.Dense(units=256)(latent)
        latent = tf.keras.layers.Activation(activation='swish')(latent)

        mu = tf.keras.layers.Dense(units=self.latent_dim)(latent)
        logvar = tf.keras.layers.Dense(units=self.latent_dim)(latent)

        outputs = Reparameterization()([mu, logvar])

        self.model = tf.keras.Model(name=self.name,
                                    inputs=[feature_inputs, mask_input],
                                    outputs=[outputs, mu, logvar])


class GeneratorModel(BaseModel):
    """
    Combines style and lexical data for image generation.
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 text_dim,
                 latent_dim,
                 blocks,
                 strides=None,
                 name='generator',
                 **kwargs):
        """
        Initializes the generator model.

        Parameters
        ----------
        image_shape : list or tuple
            Output image shape.
        lexical_shape : list or tuple
            Shape of text sequences and vocabulary.
        text_dim : int
            Text embedding size.
        latent_dim : int
            Dimension of the style latent space.
        blocks : list or tuple
            Architecture blocks (resolution).
        strides : list or tuple, optional
            List of upsampling strides (WxH).
        name : str, optional
            Model instance name.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.text_dim = text_dim
        self.latent_dim = latent_dim
        self.blocks = blocks
        self.strides = strides

        if not self.strides:
            num_blocks = len(self.blocks)
            text_length = self.lexical_shape[0] * self.lexical_shape[1]

            h_steps = int(tf.experimental.numpy.log2(self.image_shape[0] / 4))
            w_steps = int(tf.experimental.numpy.log2(self.image_shape[1] / (text_length * 4)))

            h_stride = [1] * num_blocks
            w_stride = [1] * num_blocks

            for i in range(h_steps):
                h_stride[i % num_blocks] = 2

            for i in range(w_steps):
                w_stride[i % num_blocks] = 2

            self.strides = list(zip(h_stride, w_stride))
            self.strides.reverse()

        self.build_model()

        if hasattr(self, 'model'):
            self.summary = self.model.summary
            self.call = self.model.call

    def get_config(self):
        """
        Return the configuration of the model.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'image_shape': self.image_shape,
            'lexical_shape': self.lexical_shape,
            'text_dim': self.text_dim,
            'latent_dim': self.latent_dim,
            'blocks': self.blocks,
            'strides': self.strides,
        })

        return config

    def build_model(self):
        """
        Builds the model architecture.
        """

        def residual_block(x, y, filters, up=None):
            # h = ConditionalBatchNormalization()([x, y])
            h = AdaptiveInstanceNormalization()([x, y])
            h = tf.keras.layers.Activation(activation='swish')(h)

            if up and sum(up) > 2:
                h = tf.keras.layers.UpSampling2D(size=up, interpolation='bilinear')(h)
                x = tf.keras.layers.UpSampling2D(size=up, interpolation='bilinear')(x)

            h = tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same'))(h)

            # h = ConditionalBatchNormalization()([h, y])
            h = AdaptiveInstanceNormalization()([h, y])
            h = tf.keras.layers.Activation(activation='swish')(h)

            h = tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same'))(h)

            x = tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='valid'))(x)

            return tf.keras.layers.Add()([h, x])

        #######################################
        text_input = tf.keras.layers.Input(shape=self.lexical_shape[:-1])
        text = tf.keras.layers.Flatten()(text_input)

        embedding = tf.keras.layers.Embedding(input_dim=self.lexical_shape[-1],
                                              output_dim=self.text_dim,
                                              mask_zero=True)(text)

        # embedding = tf.keras.layers.Add()([embedding, PositionEmbedding()(embedding)])
        #######################################

        #######################################
        latent_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        latent = tf.keras.layers.Identity()(latent_input)

        latent_tile = tf.keras.layers.Lambda(function=lambda x, y: tf.tile(tf.expand_dims(x, axis=1), y),
                                             arguments={'y': [1, embedding.shape[1], 1]},
                                             name='latent_tile')(latent_input)
        #######################################

        #######################################
        block = tf.keras.layers.Concatenate(axis=-1)([embedding, latent_tile])

        block = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Dense(units=4 * 4 * self.blocks[0] * 2))(block)

        # block = tf.keras.layers.Reshape(target_shape=(4, text.shape[1] * 4, -1))(block)

        block = tf.keras.layers.Reshape(target_shape=(text.shape[1] * 4, 4, 4, -1))(block)
        block = tf.keras.layers.Reshape(target_shape=(-1, text.shape[1] * 4, 4))(block)

        block = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=(0, 3, 1, 2)), name='perm')(block)
        #######################################

        #######################################
        latent_chunks = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Dense(units=self.latent_dim * len(self.blocks)))(latent)

        latent_chunks = tf.keras.layers.Lambda(function=lambda x, y: tf.split(x, num_or_size_splits=y, axis=1),
                                               arguments={'y': len(self.blocks)},
                                               name='latent_chunks')(latent_chunks)

        for i, (filters, up) in enumerate(zip(self.blocks, self.strides)):
            up = (up[0] if block.shape[1] < self.image_shape[0] else 1,
                  up[1] if block.shape[2] < self.image_shape[1] else 1)

            block = residual_block(block, latent_chunks[i], filters, up=up)

        block = GatedConv2DResidual(h=self.blocks[-1]//2)(block)

        block = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=self.blocks[-1]//2, kernel_size=3, strides=1, padding='same'))(block)

        block = tf.keras.layers.Activation(activation='swish')(block)

        outputs = tf.keras.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same'))(block)

        outputs = tf.keras.layers.Activation(activation='tanh')(outputs)
        #######################################

        self.model = tf.keras.Model(name=self.name,
                                    inputs=[text_input, latent_input],
                                    outputs=outputs)


class DiscriminatorModel(BaseModel):
    """
    Distinguishes between real and generated images.
    """

    def __init__(self,
                 image_shape,
                 blocks,
                 strides=None,
                 lexical_shape=None,
                 patch_shape=None,
                 name='discriminator',
                 **kwargs):
        """
        Initializes the discriminator model.

        Parameters
        ----------
        image_shape : list or tuple
            Input image shape.
        blocks : list or tuple
            Architecture blocks (resolution).
        strides : list or tuple
            List of downsampling strides (WxH).
        lexical_shape : list or tuple
            Shape of text sequences and vocabulary.
        patch_shape : list, tuple or None
            Patch shape, if applicable.
        name : str, optional
            Model instance name.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.image_shape = image_shape
        self.blocks = blocks
        self.strides = strides
        self.lexical_shape = lexical_shape
        self.patch_shape = patch_shape

        if not self.strides:
            self.strides = [(2, 2)] * len(self.blocks)

            if self.lexical_shape:
                num_blocks = len(self.blocks)
                text_length = self.lexical_shape[0] * self.lexical_shape[1]

                h_steps = int(tf.experimental.numpy.log2(self.image_shape[0] / 4))
                w_steps = int(tf.experimental.numpy.log2(self.image_shape[1] / (text_length * 4)))

                h_stride = [1] * num_blocks
                w_stride = [1] * num_blocks

                for i in range(h_steps):
                    h_stride[i % num_blocks] = 2

                for i in range(w_steps):
                    w_stride[i % num_blocks] = 2

                self.strides = list(zip(h_stride, w_stride))

        self.build_model()

        if hasattr(self, 'model'):
            self.summary = self.model.summary
            self.call = self.model.call

    def get_config(self):
        """
        Return the configuration of the model.

        Returns
        -------
        dict
            Configuration dictionary.
        """

        config = super().get_config()

        config.update({
            'image_shape': self.image_shape,
            'blocks': self.blocks,
            'strides': self.strides,
            'lexical_shape': self.lexical_shape,
            'patch_shape': self.patch_shape,
        })

    def build_model(self):
        """
        Builds the model architecture.
        """

        def residual_block(x, filters, preactive=True, down=None):
            h = tf.keras.layers.Identity()(x)

            if preactive:
                h = tf.keras.layers.Activation(activation='swish')(h)

            h = tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same'))(h)

            h = tf.keras.layers.Activation(activation='swish')(h)

            h = tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same'))(h)

            if preactive:
                x = tf.keras.layers.SpectralNormalization(
                    tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='valid'))(x)

            if down and sum(down) > 2:
                h = tf.keras.layers.AveragePooling2D(pool_size=2, strides=down, padding='same')(h)
                x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=down, padding='same')(x)

            if not preactive:
                x = tf.keras.layers.SpectralNormalization(
                    tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='valid'))(x)

            return tf.keras.layers.Add()([h, x])

        image_inputs = tf.keras.layers.Input(shape=self.image_shape)

        block = ExtractPatches(patch_shape=self.patch_shape)(image_inputs)

        for i, (filters, down) in enumerate(zip(self.blocks, self.strides)):
            down = (down[0] if block.shape[1] > 4 else 1,
                    down[1] if block.shape[2] > 4 else 1)

            block = residual_block(block, filters, preactive=(i > 0), down=down)

        block = residual_block(block, self.blocks[-1], preactive=True, down=None)
        block = tf.keras.layers.Activation(activation='swish')(block)

        outputs = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2]), name='reduce')(block)
        outputs = tf.keras.layers.SpectralNormalization(tf.keras.layers.Dense(units=1))(outputs)

        self.model = tf.keras.Model(name=self.name,
                                    inputs=image_inputs,
                                    outputs=outputs)
