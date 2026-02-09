import tensorflow as tf

from sarah.models.components.base import BaseModel
from sarah.models.components.base import BaseSynthesisModel
from sarah.models.components.layers import AdaptiveInstanceNormalization
from sarah.models.components.layers import ContentAlignment
from sarah.models.components.layers import ExtractPatches
from sarah.models.components.layers import GatedResidualConv2D
from sarah.models.components.layers import Reparameterization

from sarah.models.recognition.flor import RecognitionModel
from sarah.models.writer_identification.flor import WriterIdentificationModel


class SynthesisModel(BaseSynthesisModel):
    """
    Integrates multiple submodels for handwriting recognition, writer identification,
        style encoding, generation, and discrimination.

    References
    ----------
    Adversarial Generation of Handwritten Text Images Conditioned on Sequences
        https://arxiv.org/abs/1903.00277

    A Style-Based Generator Architecture for Generative Adversarial Networks
        https://arxiv.org/pdf/1812.04948

    Conditional Generative Adversarial Nets
        https://arxiv.org/abs/1411.1784

    HiGAN+: Handwriting Imitation GAN with Disentangled Representations
        https://dl.acm.org/doi/10.1145/3550070

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

        super().compile(run_eagerly=False, jit_compile=False)

        if learning_rate is None:
            learning_rate = 1e-4

        self.d_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.95, weight_decay=0.01, epsilon=1e-7)

        self.g_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.95, weight_decay=0.01, epsilon=1e-7)

        self.r_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.99, weight_decay=0.01, epsilon=1e-7)

        self.w_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.999, weight_decay=0.01, epsilon=1e-7)

    def build_model(self):
        """
        Builds the model architecture.
        """

        text_dim = 32
        latent_dim = 96

        generator_blocks = [256, 128, 64, 32]
        discriminator_blocks = [32, 64, 128, 256]

        self.discriminator = DiscriminatorModel(name='discriminator',
                                                image_shape=self.image_shape,
                                                blocks=discriminator_blocks,
                                                patch_shape=None)

        self.generator = GeneratorModel(name='generator',
                                        image_shape=self.image_shape,
                                        lexical_shape=self.lexical_shape,
                                        text_dim=text_dim,
                                        latent_dim=latent_dim,
                                        blocks=generator_blocks)

        self.recognition = RecognitionModel(name='recognition',
                                            image_shape=self.image_shape,
                                            lexical_shape=self.lexical_shape).recognition

        self.writer_encoder = WriterIdentificationModel(name='writer',
                                                        image_shape=self.image_shape,
                                                        writers_shape=self.writers_shape,
                                                        return_features=True).encoder

        self.writer_decoder = WriterIdentificationModel(name='writer',
                                                        image_shape=self.image_shape,
                                                        writers_shape=self.writers_shape,
                                                        return_features=False).decoder

        self.style_encoder = StyleEncoderModel(name='style_encoder',
                                               features_shape=self.writer_encoder.model.output[0].shape[1:],
                                               latent_dim=latent_dim)

    def train_step(self, input_data):
        """
        Executes a training step.

        Parameters
        ----------
        input_data : list or tuple
            Model inputs and targets (x_data, y_data).

        Returns
        -------
        dict
            Training metrics and losses.
        """

        self.discriminator_step(input_data)

        tf.cond(pred=tf.math.equal(tf.math.mod(self.global_step, self.generator_steps), 0),
                true_fn=lambda: self.generator_step(input_data),
                false_fn=lambda: None)

        self.global_step.assign_add(value=1)

        return self.measure_tracker.result()

    def discriminator_step(self, input_data):
        """
        Updates the discriminator.

        Parameters
        ----------
        input_data : list or tuple
            Model inputs and targets (x_data, y_data).
        """

        x_data, y_data = input_data

        aug_image_data, aug_text_data, _, aug_mask_data = x_data
        image_data, text_data, writer_data, mask_data = y_data

        self.discriminator.trainable = True
        self.recognition.trainable = True
        self.writer_encoder.trainable = True
        self.writer_decoder.trainable = True
        self.style_encoder.trainable = False
        self.generator.trainable = False

        for _ in range(self.discriminator_steps):
            random_latent_shape = (tf.shape(image_data)[0], self.style_encoder.latent_dim)
            random_latent_data = tf.random.normal(shape=random_latent_shape)

            _, real_features_data = self.writer_encoder(image_data, training=False)
            _, _, real_latent_data = self.style_encoder(real_features_data, training=False)

            real_real_images = self.generator([text_data, real_latent_data, mask_data], training=False)
            fake_real_images = self.generator([aug_text_data, real_latent_data, aug_mask_data], training=False)
            fake_fake_images = self.generator([aug_text_data, random_latent_data, aug_mask_data], training=False)
            real_fake_images = self.generator([text_data, random_latent_data, mask_data], training=False)

            real_images = tf.concat([image_data, aug_image_data], axis=0)
            fake_images = tf.concat([real_real_images, fake_real_images, fake_fake_images, real_fake_images], axis=0)

            # discriminator
            with tf.GradientTape() as d_tape:
                real_adv = self.discriminator(real_images, training=True)
                real_adv_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_adv))

                fake_adv = self.discriminator(fake_images, training=True)
                fake_adv_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_adv))

                d_adv_loss = real_adv_loss + fake_adv_loss

            d_gradients = d_tape.gradient(d_adv_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_weights))

            # handwriting recognition
            with tf.GradientTape() as r_tape:
                ctc_logits = self.recognition(aug_image_data, training=True)
                d_ctc_loss = self.ctc_loss(text_data, ctc_logits)

            r_gradients = r_tape.gradient(d_ctc_loss, self.recognition.trainable_weights)
            self.r_optimizer.apply_gradients(zip(r_gradients, self.recognition.trainable_weights))

            # writer identification
            with tf.GradientTape() as w_tape:
                _, wid_features_data = self.writer_encoder(aug_image_data, training=True)
                wid_logits = self.writer_decoder(wid_features_data, training=True)
                d_wid_loss = self.sce_loss(writer_data, wid_logits)

            w_gradients = w_tape.gradient(d_wid_loss,
                                          self.writer_encoder.trainable_weights +
                                          self.writer_decoder.trainable_weights)

            self.w_optimizer.apply_gradients(zip(w_gradients,
                                                 self.writer_encoder.trainable_weights +
                                                 self.writer_decoder.trainable_weights))

        self.measure_tracker.update({
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
            Model inputs and targets (x_data, y_data).
        """

        x_data, y_data = input_data

        _, aug_text_data, _, aug_mask_data = x_data
        image_data, text_data, writer_data, mask_data = y_data

        random_latent_shape = (tf.shape(image_data)[0], self.style_encoder.latent_dim)
        random_latent_data = tf.random.normal(shape=random_latent_shape)

        self.discriminator.trainable = False
        self.recognition.trainable = False
        self.writer_encoder.trainable = False
        self.writer_decoder.trainable = False
        self.style_encoder.trainable = True
        self.generator.trainable = True

        with tf.GradientTape(persistent=True) as g_tape:
            real_feats, real_features = self.writer_encoder(image_data, training=False)
            mu, logvar, real_latent_data = self.style_encoder(real_features, training=True)

            real_real_images = self.generator([text_data, real_latent_data, mask_data], training=True)
            fake_real_images = self.generator([aug_text_data, real_latent_data, aug_mask_data], training=True)
            fake_fake_images = self.generator([aug_text_data, random_latent_data, aug_mask_data], training=True)
            real_fake_images = self.generator([text_data, random_latent_data, mask_data], training=True)

            # discriminator
            fake_images = tf.concat([real_real_images, fake_real_images, fake_fake_images, real_fake_images], axis=0)

            fake_adv = self.discriminator(fake_images, training=False)
            g_adv_loss = -tf.reduce_mean(fake_adv)

            # handwriting recognition
            real_real_ctc = self.recognition(real_real_images, training=False)
            real_real_ctc_loss = self.ctc_loss(text_data, real_real_ctc)

            fake_real_ctc = self.recognition(fake_real_images, training=False)
            fake_real_ctc_loss = self.ctc_loss(aug_text_data, fake_real_ctc)

            fake_fake_ctc = self.recognition(fake_fake_images, training=False)
            fake_fake_ctc_loss = self.ctc_loss(aug_text_data, fake_fake_ctc)

            real_fake_ctc = self.recognition(real_fake_images, training=False)
            real_fake_ctc_loss = self.ctc_loss(text_data, real_fake_ctc)

            g_ctc_loss = real_real_ctc_loss + fake_real_ctc_loss + fake_fake_ctc_loss + real_fake_ctc_loss

            # writer identifier
            real_real_wid_feats, real_real_wid_features = self.writer_encoder(real_real_images, training=False)
            real_real_wid_logits = self.writer_decoder(real_real_wid_features, training=False)
            real_real_wid_loss = self.sce_loss(writer_data, real_real_wid_logits)

            fake_real_wid_feats, fake_real_wid_features = self.writer_encoder(fake_real_images, training=False)
            fake_real_wid_logits = self.writer_decoder(fake_real_wid_features, training=False)
            fake_real_wid_loss = self.sce_loss(writer_data, fake_real_wid_logits)

            g_wid_loss = real_real_wid_loss + fake_real_wid_loss

            # style reconstruction
            _, fake_fake_res_features = self.writer_encoder(fake_fake_images, training=False)
            _, _, fake_fake_res_data = self.style_encoder(fake_fake_res_features, training=True)
            fake_fake_res_loss = tf.reduce_mean(tf.math.square(random_latent_data - fake_fake_res_data))

            _, real_fake_res_features = self.writer_encoder(real_fake_images, training=False)
            _, _, real_fake_res_data = self.style_encoder(real_fake_res_features, training=True)
            real_fake_res_loss = tf.reduce_mean(tf.math.square(random_latent_data - real_fake_res_data))

            g_res_loss = fake_fake_res_loss + real_fake_res_loss

            # content reconstruction
            g_rec_loss = tf.reduce_mean(tf.math.square(image_data - real_real_images))

            # kl divergence
            g_kld_loss = self.kld_loss(mu, logvar)

            # contextual
            g_ctx_loss = tf.constant(0.0)

            for real_feat, real_real_feat, fake_real_feat in \
                    zip(real_feats, real_real_wid_feats, fake_real_wid_feats):

                real_real_ctx_loss = self.ctx_loss(real_feat, real_real_feat)
                fake_real_ctx_loss = self.ctx_loss(real_feat, fake_real_feat)

                g_ctx_loss += real_real_ctx_loss + fake_real_ctx_loss

            # gradient balancing
            with g_tape.stop_recording():
                grad_adv = g_tape.gradient(g_adv_loss, fake_images)
                grad_rec = g_tape.gradient(g_rec_loss, real_latent_data)
                grad_res = g_tape.gradient(g_res_loss, [fake_fake_res_data, real_fake_res_data])

                gp_adv = tf.math.reduce_std(grad_adv)
                gp_rec = tf.math.divide_no_nan(gp_adv, tf.math.reduce_std(grad_rec)) + 1
                gp_res = tf.math.divide_no_nan(gp_adv, tf.math.reduce_std(grad_res)) + 1

            # generator loss
            adv_dict = {
                'g_adv_loss': g_adv_loss,
                'g_ctx_loss': g_ctx_loss * 2,
                'g_kld_loss': g_kld_loss * 0.01,
            }

            gen_dict = {
                'g_ctc_loss': g_ctc_loss,
                'g_wid_loss': g_wid_loss,
            }

            aux_dict = {
                'g_rec_loss': g_rec_loss,
                'g_res_loss': g_res_loss,
            }

            wtd_aux_dict = {
                'g_rec_loss_w': g_rec_loss * gp_rec,
                'g_res_loss_w': g_res_loss * gp_res,
            }

            wtd_gen_dict, trainable_weights = self.measure_tracker.weight(gen_dict)

            g_loss = sum(adv_dict.values()) + sum(gen_dict.values()) + sum(aux_dict.values())
            g_loss_w = sum(adv_dict.values()) + sum(wtd_gen_dict.values()) + sum(wtd_aux_dict.values())

        g_gradients = g_tape.gradient(g_loss_w,
                                      self.style_encoder.trainable_weights +
                                      self.generator.trainable_weights +
                                      trainable_weights)

        self.g_optimizer.apply_gradients(zip(g_gradients,
                                             self.style_encoder.trainable_weights +
                                             self.generator.trainable_weights +
                                             trainable_weights))
        del g_tape

        # kid
        self.kid.update_state(image_data, real_real_images)

        self.measure_tracker.update({
            **adv_dict,
            **gen_dict,
            **aux_dict,
            **wtd_gen_dict,
            **wtd_aux_dict,
            'loss': g_loss,
            'loss_w': g_loss_w,
            self.kid.name: self.kid.result(),
        })


class StyleEncoderModel(BaseModel):
    """
    Encodes style features from images into a latent style vector.
    """

    def __init__(self,
                 features_shape,
                 latent_dim,
                 name='style_encoder',
                 **kwargs):
        """
        Initializes the style encoder model.

        Parameters
        ----------
        features_shape : list or tuple
            Input feature shape.
        latent_dim : int
            Dimension of the style latent space.
        name : str, optional
            Model instance name.
        **kwargs : dict
            Additional arguments.
        """

        super().__init__(name=name, **kwargs)

        self.features_shape = features_shape
        self.latent_dim = latent_dim

        self.build_model()

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
            'latent_dim': self.latent_dim,
        })

    def build_model(self):
        """
        Builds the model architecture.
        """

        feature_inputs = tf.keras.layers.Input(shape=self.features_shape)

        style = tf.keras.layers.Dense(units=256)(feature_inputs)
        style = tf.keras.layers.Activation(activation='swish')(style)

        style = tf.keras.layers.Dense(units=256)(style)
        style = tf.keras.layers.Activation(activation='swish')(style)

        mu = tf.keras.layers.Dense(units=self.latent_dim)(style)
        logvar = tf.keras.layers.Dense(units=self.latent_dim)(style)

        outputs = Reparameterization()([mu, logvar])

        self.model = tf.keras.Model(name=self.name,
                                    inputs=feature_inputs,
                                    outputs=[mu, logvar, outputs])


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

        self.num_blocks = len(self.blocks)
        self.nonlocal_size = (self.image_shape[0] * self.image_shape[1]) / 4

        self.base_patch = (4, 4)
        self.base_shape = (self.lexical_shape[0] * self.base_patch[0],
                           self.lexical_shape[1] * self.base_patch[1])

        if not self.strides:
            h_steps = int(tf.keras.ops.log2(self.image_shape[0] / self.base_shape[0]))
            w_steps = int(tf.keras.ops.log2(self.image_shape[1] / self.base_shape[1]))

            h_stride = [1] * self.num_blocks
            w_stride = [1] * self.num_blocks

            for i in range(h_steps):
                h_stride[i % self.num_blocks] = 2

            for i in range(w_steps):
                w_stride[i % self.num_blocks] = 2

            self.strides = list(zip(h_stride, w_stride))
            self.strides.reverse()

        self.build_model()

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
            h = tf.keras.layers.Identity()(x)

            h = AdaptiveInstanceNormalization(epsilon=1e-3)([h, y])
            h = tf.keras.layers.Activation(activation='swish')(h)

            if up and sum(up) > 2:
                h = tf.keras.layers.UpSampling2D(size=up, interpolation='bicubic')(h)
                x = tf.keras.layers.UpSampling2D(size=up, interpolation='bicubic')(x)

            h = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(h)

            h = AdaptiveInstanceNormalization(epsilon=1e-3)([h, y])
            h = tf.keras.layers.Activation(activation='swish')(h)

            h = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(h)
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='valid')(x)

            return tf.keras.layers.Add()([h, x])

        mask_input = tf.keras.layers.Input(shape=self.image_shape)
        mask = tf.keras.layers.Identity()(mask_input)

        latent_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        latent = tf.keras.layers.Identity()(latent_input)

        text_input = tf.keras.layers.Input(shape=self.lexical_shape[:-1])
        text = tf.keras.layers.Flatten()(text_input)

        embedding = tf.keras.layers.Embedding(input_dim=self.lexical_shape[-1],
                                              output_dim=self.text_dim)(text)

        latent_tile = tf.keras.layers.Lambda(function=lambda x, y: tf.tile(tf.expand_dims(x, axis=1), y),
                                             arguments={'y': [1, embedding.shape[1], 1]},
                                             name='latent_tile')(latent)

        embedding = tf.keras.layers.Concatenate(axis=-1)([embedding, latent_tile])
        embedding = tf.keras.layers.LayerNormalization(epsilon=1e-3)(embedding)

        block = tf.keras.layers.Dense(units=self.base_patch[0] * self.base_patch[1] * self.blocks[0] * 2)(embedding)

        block = tf.keras.layers.Reshape(target_shape=(self.base_shape[1], self.base_shape[0], -1))(block)
        block = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=(0, 2, 1, 3)), name='perm')(block)

        latent_chunks = tf.keras.layers.Dense(units=self.latent_dim * self.num_blocks)(latent)

        latent_chunks = tf.keras.layers.Lambda(function=lambda x, y: tf.split(x, num_or_size_splits=y, axis=1),
                                               arguments={'y': self.num_blocks},
                                               name='latent_chunks')(latent_chunks)

        for i, (filters, up) in enumerate(zip(self.blocks, self.strides)):
            up = (up[0] if block.shape[1] < self.image_shape[0] * 2 else 1,
                  up[1] if block.shape[2] < self.image_shape[1] * 2 else 1)

            if block.shape[1] * block.shape[2] == self.nonlocal_size:
                block = GatedResidualConv2D()(block)

            block = residual_block(block, latent_chunks[i], filters, up=up)

        block = residual_block(block, latent, self.blocks[-1] // 2, up=(1, 2))
        block = tf.keras.layers.Activation(activation='swish')(block)

        block = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='valid')(block)
        block = tf.keras.layers.Activation(activation='tanh')(block)

        outputs = ContentAlignment(char_height_ratio=block.shape[1] // self.lexical_shape[0],
                                   char_width_ratio=block.shape[2] // self.lexical_shape[1])([block, text, mask])

        self.model = tf.keras.Model(name=self.name,
                                    inputs=[text_input, latent_input, mask_input],
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

        self.num_blocks = len(self.blocks)
        self.nonlocal_size = (self.image_shape[0] * self.image_shape[1]) / 4

        self.base_patch = (4, 4)

        if not self.strides:
            self.strides = [(2, 2)] * self.num_blocks

            if self.lexical_shape:
                self.base_shape = (self.lexical_shape[0] * self.base_patch[0],
                                   self.lexical_shape[1] * self.base_patch[1])

                h_steps = int(tf.keras.ops.log2(self.image_shape[0] / self.base_shape[0]))
                w_steps = int(tf.keras.ops.log2(self.image_shape[1] / self.base_shape[1]))

                h_stride = [1] * self.num_blocks
                w_stride = [1] * self.num_blocks

                for i in range(h_steps):
                    h_stride[i % self.num_blocks] = 2

                for i in range(w_steps):
                    w_stride[i % self.num_blocks] = 2

                self.strides = list(zip(h_stride, w_stride))

        self.build_model()

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

            h = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same')(h)

            h = tf.keras.layers.Activation(activation='swish')(h)
            h = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same')(h)

            if preactive:
                x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='valid')(x)

            if down and sum(down) > 2:
                h = tf.keras.layers.AveragePooling2D(pool_size=2, strides=down, padding='same')(h)
                x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=down, padding='same')(x)

            if not preactive:
                x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='valid')(x)

            return tf.keras.layers.Add()([h, x])

        image_input = tf.keras.layers.Input(shape=self.image_shape)

        block = ExtractPatches(patch_shape=self.patch_shape, strides=(2, 2), padding='valid')(image_input)

        for i, (filters, down) in enumerate(zip(self.blocks, self.strides)):
            down = (down[0] if block.shape[1] > self.base_patch[0] else 1,
                    down[1] if block.shape[2] > self.base_patch[1] else 1)

            block = residual_block(block, filters, preactive=(i > 0), down=down)

            if block.shape[1] * block.shape[2] == self.nonlocal_size:
                block = GatedResidualConv2D()(block)

        if not self.patch_shape:
            block = residual_block(block, self.blocks[-1], preactive=True, down=None)

        block = tf.keras.layers.Activation(activation='swish')(block)
        block = tf.keras.layers.GlobalAveragePooling2D()(block)

        outputs = tf.keras.layers.Dense(units=1)(block)

        self.model = tf.keras.Model(name=self.name,
                                    inputs=image_input,
                                    outputs=outputs)
