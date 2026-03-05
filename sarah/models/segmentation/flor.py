import tensorflow as tf

from sarah.models.components.base import BaseSegmentationModel


class SegmentationModel(BaseSegmentationModel):
    """
    TensorFlow model for handwriting image segmentation.
    Features a lightweight CNN for binary segmentation.

    References
    ----------
    Processamento digital de imagens para detecção automática de fissuras em revestimentos cerâmicos de edifícios
        https://www.scielo.br/j/ac/a/fkNKmmBtpzy9LsB7sg5fbBm

    U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
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

        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.999, weight_decay=0.01, epsilon=1e-7)

    def build_model(self):
        """
        Builds the model architecture.
        """

        feats = []

        # encoder model
        encoder_input = tf.keras.Input(shape=self.image_shape)

        conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same')(encoder_input)
        conv1 = tf.keras.layers.Activation(activation='swish')(conv1)
        conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same')(conv1)
        conv1 = tf.keras.layers.Activation(activation='swish')(conv1)
        feats.append(conv1)
        conv1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 4), strides=(2, 4))(conv1)

        conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(conv1)
        conv2 = tf.keras.layers.Activation(activation='swish')(conv2)
        conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(conv2)
        conv2 = tf.keras.layers.Activation(activation='swish')(conv2)
        feats.append(conv2)
        conv2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 4), strides=(2, 4))(conv2)

        conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(conv2)
        conv3 = tf.keras.layers.Activation(activation='swish')(conv3)
        conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(conv3)
        conv3 = tf.keras.layers.Activation(activation='swish')(conv3)
        feats.append(conv3)
        conv3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 4), strides=(2, 4))(conv3)

        conv4 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same')(conv3)
        conv4 = tf.keras.layers.Activation(activation='swish')(conv4)
        conv4 = tf.keras.layers.Dropout(rate=0.3)(conv4)
        conv4 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same')(conv4)
        conv4 = tf.keras.layers.Activation(activation='swish')(conv4)
        feats.append(conv4)
        conv4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 4), strides=(2, 4))(conv4)

        encoder = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(conv4)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        encoder = tf.keras.layers.Dropout(rate=0.3)(encoder)
        encoder = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')(encoder)
        encoder = tf.keras.layers.Activation(activation='swish')(encoder)
        feats.append(encoder)

        self.encoder = tf.keras.Model(name='segmentation_encoder', inputs=encoder_input, outputs=feats)

        # decoder model
        decoder_input = [tf.keras.Input(shape=x.shape[1:]) for x in feats]

        conv6 = tf.keras.layers.UpSampling2D(size=(2, 4), interpolation='bilinear')(decoder_input[-1])
        conv6 = tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same')(conv6)
        conv6 = tf.keras.layers.Activation(activation='swish')(conv6)

        conv6 = tf.keras.layers.Concatenate(axis=-1)([decoder_input[3], conv6])

        conv6 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same')(conv6)
        conv6 = tf.keras.layers.Activation(activation='swish')(conv6)
        conv6 = tf.keras.layers.Dropout(rate=0.3)(conv6)
        conv6 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same')(conv6)
        conv6 = tf.keras.layers.Activation(activation='swish')(conv6)

        conv7 = tf.keras.layers.UpSampling2D(size=(2, 4), interpolation='bilinear')(conv6)
        conv7 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same')(conv7)
        conv7 = tf.keras.layers.Activation(activation='swish')(conv7)

        conv7 = tf.keras.layers.Concatenate(axis=-1)([decoder_input[2], conv7])

        conv7 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(conv7)
        conv7 = tf.keras.layers.Activation(activation='swish')(conv7)
        conv7 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(conv7)
        conv7 = tf.keras.layers.Activation(activation='swish')(conv7)

        conv8 = tf.keras.layers.UpSampling2D(size=(2, 4), interpolation='bilinear')(conv7)
        conv8 = tf.keras.layers.Conv2D(filters=16, kernel_size=2, padding='same')(conv8)
        conv8 = tf.keras.layers.Activation(activation='swish')(conv8)

        conv8 = tf.keras.layers.Concatenate(axis=-1)([decoder_input[1], conv8])

        conv8 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(conv8)
        conv8 = tf.keras.layers.Activation(activation='swish')(conv8)
        conv8 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same')(conv8)
        conv8 = tf.keras.layers.Activation(activation='swish')(conv8)

        conv9 = tf.keras.layers.UpSampling2D(size=(2, 4), interpolation='bilinear')(conv8)
        conv9 = tf.keras.layers.Conv2D(filters=8, kernel_size=2, padding='same')(conv9)
        conv9 = tf.keras.layers.Activation(activation='swish')(conv9)

        conv9 = tf.keras.layers.Concatenate(axis=-1)([decoder_input[0], conv9])

        conv9 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same')(conv9)
        conv9 = tf.keras.layers.Activation(activation='swish')(conv9)
        conv9 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same')(conv9)
        conv9 = tf.keras.layers.Activation(activation='swish')(conv9)

        decoder = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same')(conv9)
        decoder = tf.keras.layers.Activation(activation='sigmoid')(decoder)

        self.decoder = tf.keras.Model(name='segmentation_decoder', inputs=decoder_input, outputs=decoder)

        # segmentation model
        if self.return_features:
            encoder_output = self.encoder(encoder_input)
            outputs = [encoder_output, self.decoder(encoder_output)]
        else:
            outputs = self.decoder(self.encoder(encoder_input))

        self.segmentation = tf.keras.Model(name=self.name,
                                           inputs=self.encoder.input,
                                           outputs=outputs)
