import tensorflow as tf

from models.components.attention import SpectralSelfAttention
from models.components.loss import CTCLoss
from models.components.loss import CTXLoss
from models.components.loss import L1Loss
from models.components.metric import KID
from models.components.normalization import ConditionalBatchNormalization
from models.components.normalization import SpectralNormalization
from models.components.optimizer import NormalizedOptimizer
from models.components.processing import AdaptiveDenseReshape
from models.components.processing import ExtractPatches


class HandwritingSynthesis(tf.keras.Model):
    """
    A comprehensive synthesis model built on the TensorFlow Keras framework.

    This model integrates several sub-models including a generator, discriminator,
        patch discriminator, style backbone, style encoder, writer identifier, and text recognizer.
    Each of these components is specialized for different aspects of image and text processing,
        and they work together to enable complex synthesis and recognition tasks.

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
        """
        Initialize the synthesis model with specified parameters for each sub-model.

        Parameters
        ----------
        image_shape : tuple or list
            The shape of the input images.
        patch_shape : tuple or list
            The shape of patches for the patch discriminator.
        lexical_shape : tuple or list
            The shape of the lexical input.
        writer_dim : int
            The dimension for the writer identifier.
        latent_dim : int
            The latent dimension size for the style encoder.
        embedding_dim : int
            The embedding dimension size.
        backbone_blocks : tuple or list
            The blocks of filters for the style backbone model.
        generator_blocks : tuple or list
            The blocks of filters for the generator model.
        discriminator_blocks : tuple or list
            The blocks of filters for the discriminator models.
        **kwargs : dict
            Additional keyword arguments for the TensorFlow Keras Model.
        """

        super().__init__(**kwargs)

        self.generator = Generator(image_shape=image_shape,
                                   lexical_shape=lexical_shape,
                                   latent_dim=latent_dim,
                                   embedding_dim=embedding_dim,
                                   blocks=generator_blocks,
                                   name='generator')

        self.discriminator = Discriminator(image_shape=image_shape,
                                           patch_shape=None,
                                           embedding_dim=embedding_dim,
                                           blocks=discriminator_blocks,
                                           name='discriminator')

        self.patch_discriminator = Discriminator(image_shape=image_shape,
                                                 patch_shape=patch_shape,
                                                 embedding_dim=embedding_dim,
                                                 blocks=discriminator_blocks,
                                                 name='patch_discriminator')

        self.style_backbone = StyleBackbone(image_shape=image_shape,
                                            blocks=backbone_blocks,
                                            name='style_backbone')

        self.style_encoder = StyleEncoder(features_shape=self.style_backbone.features_shape,
                                          latent_dim=latent_dim,
                                          name='style_encoder')

        self.writer_identification = WriterIdentification(features_shape=self.style_backbone.features_shape,
                                                          writer_dim=writer_dim,
                                                          name='writer_identification')

        self.handwriting_recognition = HandwritingRecognition(image_shape=image_shape,
                                                              lexical_shape=lexical_shape,
                                                              blocks=backbone_blocks,
                                                              name='handwriting_recognition')

        self.names = [
            self.generator.name,
            self.discriminator.name,
            self.patch_discriminator.name,
            self.style_backbone.name,
            self.style_encoder.name,
            self.writer_identification.name,
            self.handwriting_recognition.name,
        ]

    def __repr__(self):
        """
        Provides a formatted string with useful information.

        Returns
        -------
        str
            Formatted string with useful information.
        """

        info = "=================================================="
        info += f"\n{self.__class__.__name__.center(50)}"

        for name in self.names:
            if not hasattr(self, name):
                continue

            model = getattr(self, name)

            trainable_count = sum([tf.size(x).numpy() for x in model.trainable_variables])
            non_trainable_count = sum([tf.size(x).numpy() for x in model.non_trainable_variables])
            total_count = trainable_count + non_trainable_count

            info += "\n--------------------------------------------------"
            info += f"\n{'Model':<{25}}: {model.name}"
            info += "\n--------------------------------------------------"
            info += f"\n{'Total params':<{25}}: {total_count:,}"
            info += f"\n{'Trainable params':<{25}}: {trainable_count:,}"
            info += f"\n{'Non-trainable params':<{25}}: {non_trainable_count:,}"
            info += f"\n{'Size (MB)':<{25}}: {(total_count*4) / (1024**2):,.2f}"

        return info

    def summary(self):
        """
        Print a summary of each sub-model.

        This method provides information about total, trainable, and non-trainable parameters,
            as well as the size in MB of each sub-model.
        """

        for name in self.names:
            if not hasattr(self, name):
                continue

            model = getattr(self, name)
            model.summary()

    def get_weights(self):
        """
        Retrieve the weights of the sub-models.

        Returns
        -------
        dict
            A dictionary with sub-model names as keys and their weights as values.
        """

        with self.distribute_strategy.scope():
            weights = {}

            for name in self.names:
                if getattr(self, name) is None:
                    continue

                weights[name] = getattr(self, name).get_weights()

            return weights

    def set_weights(self, weights):
        """
        Set the weights for the sub-models.

        Parameters
        ----------
        weights : dict
            A dictionary with sub-model names as keys and their weights as values.
        """

        for name in self.names:
            if getattr(self, name) is None:
                continue

            getattr(self, name).set_weights(weights[name])

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """
        Save the weights of the sub-models.

        Parameters
        ----------
        filepath : str
            Filepath for saving the weights.
        overwrite : bool, optional
            Whether to overwrite the existing file.
        save_format : str, optional
            Format of the file to save the weights.
        options : tf.train.CheckpointOptions, optional
            Optional arguments to pass to tf.train.Checkpoint.save.
        """

        for name in self.names:
            if getattr(self, name) is None:
                continue

            getattr(self, name).save_weights(filepath=filepath.replace('model', name),
                                             overwrite=overwrite,
                                             save_format=save_format,
                                             options=options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        """
        Load the weights for the sub-models.

        Parameters
        ----------
        filepath : str
            Filepath for loading the weights.
        by_name : bool, optional
            Load weights by name.
        skip_mismatch : bool, optional
            Skip loading of layers where there is a mismatch in the number of weights.
        options : tf.train.CheckpointOptions, optional
            Optional arguments to pass to tf.train.Checkpoint.load.
        """

        for name in self.names:
            if getattr(self, name) is None:
                continue

            getattr(self, name).load_weights(filepath=filepath.replace('model', name),
                                             by_name=by_name,
                                             skip_mismatch=skip_mismatch,
                                             options=options)

    def compile(self, learning_rate=0.001):
        """
        Configure the sub-models for training.

        This method sets up the optimizers, loss functions, and metrics for the model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        """

        super().compile(run_eagerly=False)

        self.g_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999))

        self.d_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999))

        self.p_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999))

        self.b_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999))

        self.e_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999))

        self.w_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999))

        self.r_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999))

        self.l1_loss = L1Loss()
        self.ctx_loss = CTXLoss()
        self.ctc_loss = CTCLoss()
        self.kld_loss = tf.keras.losses.KLDivergence()
        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.kid_metric = KID()

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

        (image_inputs, text_inputs, aug_image_inputs, aug_text_inputs, writer_inputs), _ = input_data

        batch_size = tf.shape(image_inputs)[0]
        batch_quarter = tf.math.maximum(1, batch_size // 4)

        # discriminator phase
        for i in range(4):
            q_start_index = i * batch_quarter
            q_aug_text_inputs = aug_text_inputs[q_start_index:q_start_index + batch_quarter]
            q_image_inputs = image_inputs[q_start_index:q_start_index + batch_quarter]
            q_text_inputs = text_inputs[q_start_index:q_start_index + batch_quarter]

            m_start_index = (i // 2) * batch_quarter
            m_image_inputs = image_inputs[m_start_index:m_start_index + (batch_quarter * 2)]
            m_aug_image_inputs = aug_image_inputs[m_start_index:m_start_index + (batch_quarter * 2)]

            fake_latent_inputs = tf.random.normal((batch_quarter, self.style_encoder.latent_dim), mean=0.0, stddev=1.0)

            with tf.GradientTape() as d_tape, \
                    tf.GradientTape() as p_tape, \
                    tf.GradientTape() as w_tape, \
                    tf.GradientTape() as r_tape:

                real_features_inputs, _ = self.style_backbone(q_image_inputs, training=True)
                real_latent_inputs, _, _ = self.style_encoder(real_features_inputs, training=True)

                fake_fake_images = self.generator([fake_latent_inputs, q_aug_text_inputs], training=True)
                real_fake_images = self.generator([real_latent_inputs, q_aug_text_inputs], training=True)
                real_real_images = self.generator([real_latent_inputs, q_text_inputs], training=True)
                fake_real_images = self.generator([fake_latent_inputs, q_text_inputs], training=True)

                fake_image_inputs = tf.random.shuffle(tf.concat([fake_fake_images,
                                                                 real_fake_images,
                                                                 real_real_images,
                                                                 fake_real_images], axis=0))

                real_image_inputs = tf.random.shuffle(tf.concat([m_image_inputs,
                                                                 m_aug_image_inputs], axis=0))

                # patch discriminator loss
                fake_patch_disc = self.patch_discriminator(fake_image_inputs, training=True)
                fake_patch_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_patch_disc))

                real_patch_disc = self.patch_discriminator(real_image_inputs, training=True)
                real_patch_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_patch_disc))

                p_loss = fake_patch_disc_loss + real_patch_disc_loss

                # discriminator loss
                fake_disc = self.discriminator(fake_image_inputs, training=True)
                fake_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_disc))

                real_disc = self.discriminator(real_image_inputs, training=True)
                real_disc_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_disc))

                d_loss = fake_disc_loss + fake_patch_disc_loss + real_disc_loss + real_patch_disc_loss

                # writer identifier loss
                aug_features_inputs, _ = self.style_backbone(aug_image_inputs, training=True)
                wid_logits = self.writer_identification(aug_features_inputs, training=True)
                w_loss = self.cls_loss(writer_inputs, wid_logits)

                # recognizer loss
                aug_ctc_logits = self.handwriting_recognition(aug_image_inputs, training=True)
                r_loss = self.ctc_loss(text_inputs, aug_ctc_logits)

            d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_weights))

            p_gradients = p_tape.gradient(p_loss, self.patch_discriminator.trainable_weights)
            self.p_optimizer.apply_gradients(zip(p_gradients, self.patch_discriminator.trainable_weights))

            w_gradients = w_tape.gradient(w_loss, self.writer_identification.trainable_weights)
            self.w_optimizer.apply_gradients(zip(w_gradients, self.writer_identification.trainable_weights))

            r_gradients = r_tape.gradient(r_loss, self.handwriting_recognition.trainable_weights)
            self.r_optimizer.apply_gradients(zip(r_gradients, self.handwriting_recognition.trainable_weights))

        # generator phase
        indices = tf.random.shuffle(tf.range(batch_size))

        m_image_inputs = tf.gather(image_inputs, indices[:batch_quarter])
        m_text_inputs = tf.gather(text_inputs, indices[:batch_quarter])
        m_aug_text_inputs = tf.gather(aug_text_inputs, indices[:batch_quarter])
        m_writer_inputs = tf.gather(writer_inputs, indices[:batch_quarter])

        fake_latent_inputs = tf.random.normal((batch_quarter, self.style_encoder.latent_dim), mean=0.0, stddev=1.0)

        with tf.GradientTape() as g_tape, \
                tf.GradientTape() as b_tape, \
                tf.GradientTape() as e_tape:

            real_features_inputs, real_image_feats = self.style_backbone(m_image_inputs, training=True)
            real_latent_inputs, mu, logvar = self.style_encoder(real_features_inputs, training=True)

            fake_fake_images = self.generator([fake_latent_inputs, m_aug_text_inputs], training=True)
            real_fake_images = self.generator([real_latent_inputs, m_aug_text_inputs], training=True)
            real_real_images = self.generator([real_latent_inputs, m_text_inputs], training=True)
            fake_real_images = self.generator([fake_latent_inputs, m_text_inputs], training=True)

            fake_image_inputs = tf.random.shuffle(tf.concat([fake_fake_images,
                                                             real_fake_images,
                                                             real_real_images,
                                                             fake_real_images], axis=0))

            # discriminator loss
            fake_disc = self.discriminator(fake_image_inputs, training=True)
            fake_disc_loss = -tf.reduce_mean(fake_disc)

            fake_patch_disc = self.patch_discriminator(fake_image_inputs, training=True)
            fake_patch_disc_loss = -tf.reduce_mean(fake_patch_disc)

            disc_loss = fake_disc_loss + fake_patch_disc_loss

            # ctc loss
            fake_fake_ctc_logits = self.handwriting_recognition(fake_fake_images, training=True)
            fake_fake_ctc_loss = self.ctc_loss(m_aug_text_inputs, fake_fake_ctc_logits)

            real_fake_ctc_logits = self.handwriting_recognition(real_fake_images, training=True)
            real_fake_ctc_loss = self.ctc_loss(m_aug_text_inputs, real_fake_ctc_logits)

            real_real_ctc_logits = self.handwriting_recognition(real_real_images, training=True)
            real_real_ctc_loss = self.ctc_loss(m_text_inputs, real_real_ctc_logits)

            fake_real_ctc_logits = self.handwriting_recognition(fake_real_images, training=True)
            fake_real_ctc_loss = self.ctc_loss(m_text_inputs, fake_real_ctc_logits)

            ctc_loss = fake_fake_ctc_loss + real_fake_ctc_loss + real_real_ctc_loss + fake_real_ctc_loss

            # style reconstruction loss
            fake_fake_features_inputs, _ = self.style_backbone(fake_fake_images, training=True)
            fake_fake_latent_inputs, _, _ = self.style_encoder(fake_fake_features_inputs, training=True)
            fake_fake_info_loss = tf.reduce_mean(tf.math.abs(fake_fake_latent_inputs - fake_latent_inputs))

            fake_real_features_inputs, _ = self.style_backbone(fake_real_images, training=True)
            fake_real_latent_inputs, _, _ = self.style_encoder(fake_real_features_inputs, training=True)
            fake_real_info_loss = tf.reduce_mean(tf.math.abs(fake_real_latent_inputs - fake_latent_inputs))

            info_loss = tf.reduce_mean([fake_fake_info_loss, fake_real_info_loss])

            # content restruction loss
            real_real_l1_loss = self.l1_loss(m_image_inputs, real_real_images)

            l1_loss = tf.reduce_mean(real_real_l1_loss)

            # writer identify loss
            real_fake_features_inputs, real_fake_image_feats = self.style_backbone(real_fake_images, training=True)
            real_fake_wid_logits = self.writer_identification(real_fake_features_inputs, training=True)
            real_fake_wid_loss = self.cls_loss(m_writer_inputs, real_fake_wid_logits)

            real_real_features_inputs, real_real_image_feats = self.style_backbone(real_real_images, training=True)
            real_real_wid_logits = self.writer_identification(real_real_features_inputs, training=True)
            real_real_wid_loss = self.cls_loss(m_writer_inputs, real_real_wid_logits)

            wid_loss = tf.reduce_mean([real_fake_wid_loss, real_real_wid_loss])

            # contextual loss
            ctx_loss = tf.constant(0.0)

            for real_image_feat, real_fake_image_feat, real_real_image_feat \
                    in zip(real_image_feats, real_fake_image_feats, real_real_image_feats):
                ctx_loss += self.ctx_loss(real_image_feat, real_fake_image_feat)
                ctx_loss += self.ctx_loss(real_image_feat, real_real_image_feat)

            # kl-divergency loss
            kl_loss = self.kld_loss(mu, logvar)

            # generator loss
            g_loss = (disc_loss + ctc_loss + info_loss + l1_loss + wid_loss + (ctx_loss * 5.0) + (kl_loss * 1e-4))

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_weights))

        b_gradients = b_tape.gradient(g_loss, self.style_backbone.trainable_weights)
        self.b_optimizer.apply_gradients(zip(b_gradients, self.style_backbone.trainable_weights))

        e_gradients = e_tape.gradient(g_loss, self.style_encoder.trainable_weights)
        self.e_optimizer.apply_gradients(zip(e_gradients, self.style_encoder.trainable_weights))

        # metric phase
        metrics = self.test_step(input_data)

        return {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'w_loss': w_loss,
            'r_loss': r_loss,
            **metrics,
        }

    def test_step(self, input_data):
        """
        Perform the testing step on the provided batch of data.

        Parameters
        ----------
        input_data : list or tuple
            A batch of data (x_data, y_data).

        Returns
        -------
        dict
            A dictionary containing evaluation metrics.
        """

        x_data, _ = input_data

        generated_images = self.call(x_data, training=False)
        self.kid_metric.update_state(x_data[0], generated_images)

        return {
            'kernel_inception_distance': self.kid_metric.result(),
        }

    def call(self, x_data, training=None):
        """
        Processes input images and text through the style backbone, encoder,
            and generator to produce generated images.

        Parameters
        ----------
        input_data : list or tuple
            A batch of data (x_data).
        training : bool, optional
            Indicates whether the call is for training or inference.

        Returns
        -------
        tf.Tensor
            The generated images.
        """

        image_inputs, text_inputs, _, _, _ = x_data

        features_inputs, _ = self.style_backbone(image_inputs, training=training)
        latent_inputs, _, _ = self.style_encoder(features_inputs, training=training)
        generated_images = self.generator([latent_inputs, text_inputs], training=training)

        return generated_images


class Generator(tf.keras.Model):
    """
    A generator model that combines latent and vocabulary data for generative tasks.

    This model synthesizes images based on the combination of latent space vectors and
        lexical information, suitable for tasks like image generation from textual descriptions.
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 latent_dim,
                 embedding_dim,
                 blocks,
                 **kwargs):
        """
        Initialize the generator model with specified parameters.

        Parameters
        ----------
        image_shape : list or tuple
            Shape of the output image.
        lexical_shape : list or tuple
            Shape of the text sequences and vocabulary encoding.
        latent_dim : int
            Dimension of the latent space.
        embedding_dim : int
            Dimension of the embedding space.
        blocks : list or tuple
            Blocks of channels for the model's architecture.
        **kwargs : dict
            Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.latent_dim = latent_dim
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
            'image_shape': self.image_shape,
            'lexical_shape': self.lexical_shape,
            'latent_dim': self.latent_dim,
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

        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim,))

        latent_dense = SpectralNormalization(
            tf.keras.layers.Dense(units=self.latent_dim * len(self.blocks)))(latent_inputs)

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


class Discriminator(tf.keras.Model):
    """
    A discriminator model that evaluates the authenticity of generated images.

    This model is designed to distinguish between real and synthesized images,
        typically used in GAN architectures for generative tasks.
    """

    def __init__(self,
                 image_shape,
                 patch_shape,
                 embedding_dim,
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
        embedding_dim : int
            Dimension of the embedding space.
        blocks : list or tuple
            Blocks of channels for the model's architecture.
        **kwargs : dict
            Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.patch_shape = patch_shape
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
            'image_shape': self.image_shape,
            'patch_shape': self.patch_shape,
            'embedding_dim': self.embedding_dim,
            'blocks': self.blocks,
        })

    def build_model(self):
        """
        Initializes and builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
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


class StyleBackbone(tf.keras.Model):
    """
    A backbone model that extracts style patterns from images.

    This model is designed to capture the style characteristics from input images,
        providing a basis for further style-related processing in generative tasks.
    """

    def __init__(self,
                 image_shape,
                 blocks,
                 **kwargs):
        """
        Initialize the style backbone model with specified parameters.

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
            conv = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(conv)

            feats.append(conv)

        conv = tf.keras.layers.ReLU()(conv)
        conv = tf.keras.layers.Conv2D(blocks[-1], kernel_size=3, strides=1, padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.ReLU()(conv)

        outputs = tf.keras.layers.Reshape(target_shape=(conv.get_shape()[1] * conv.get_shape()[2], -1))(conv)

        self.features_shape = outputs.get_shape()[1:]
        self.model = tf.keras.Model(inputs=image_inputs, outputs=[outputs, feats], name=self.name)


class StyleEncoder(tf.keras.Model):
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

        self.model = tf.keras.Model(inputs=feature_inputs, outputs=[outputs, mu, logvar], name=self.name)


class WriterIdentification(tf.keras.Model):
    """
    A writer identifier model that classifies handwriting images based on extracted style features.

    This model is designed to identify the writer of a given handwriting sample
        by analyzing the stylistic features of the handwriting.
    """

    def __init__(self,
                 features_shape,
                 writer_dim,
                 **kwargs):
        """
        Initialize the writer identification model with specified parameters.

        Parameters
        ----------
        features_shape : list or tuple
            Shape of the input features.
        writer_dim : int
            Number of writers to classify.
        **kwargs : dict
            Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.features_shape = features_shape
        self.writer_dim = writer_dim

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
            'writer_dim': self.writer_dim,
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

        outputs = tf.keras.layers.Dense(self.writer_dim)(style_dense)

        self.model = tf.keras.Model(inputs=feature_inputs, outputs=outputs, name=self.name)


class HandwritingRecognition(tf.keras.Model):
    """
    A recognizer model that transcribes handwriting text from images.

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
            Shape of the input image.
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
            conv = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(conv)

        conv = tf.keras.layers.ReLU()(conv)
        conv = tf.keras.layers.Conv2D(blocks[-1], kernel_size=3, strides=1, padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.ReLU()(conv)

        conv = AdaptiveDenseReshape(target_shape=self.lexical_shape)(conv)

        bgru = tf.keras.layers.Reshape(target_shape=(-1, conv.get_shape()[-1]))(conv)

        bgru = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5))(bgru)
        bgru = tf.keras.layers.Dense(units=256)(bgru)

        bgru = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5))(bgru)
        bgru = tf.keras.layers.Dense(units=self.lexical_shape[-1], activation='softmax')(bgru)

        outputs = tf.keras.layers.Reshape(target_shape=self.lexical_shape)(bgru)

        self.model = tf.keras.Model(inputs=image_inputs, outputs=outputs, name=self.name)
