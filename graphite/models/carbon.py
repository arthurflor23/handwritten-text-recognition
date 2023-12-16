import random
import tensorflow as tf

from models.components.loss import CTCLoss
from models.components.loss import CTXLoss
from models.components.loss import L1Loss
from models.components.metric import EditDistance
from models.components.metric import KernelInceptionDistance
from models.components.optimizer import NormalizedOptimizer


class BaseModel(tf.keras.Model):

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

        for name in self.model_names:
            model = getattr(self, name, None)

            if model is None:
                continue

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

    def get_weights(self):
        """
        Retrieve the weights of the submodels.

        Returns
        -------
        dict
            A dictionary with submodel names as keys and their weights as values.
        """

        with self.distribute_strategy.scope():
            weights = {}

            for name in self.model_names:
                model = getattr(self, name, None)

                if model is not None:
                    weights[name] = model.get_weights()

            return weights

    def set_weights(self, weights):
        """
        Set the weights for the submodels.

        Parameters
        ----------
        weights : dict
            A dictionary with submodel names as keys and their weights as values.
        """

        for name in self.model_names:
            model = getattr(self, name, None)

            if model is not None:
                model.set_weights(weights[name])

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """
        Save the weights of the submodels.

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

        for name in self.model_names:
            model = getattr(self, name, None)

            if model is not None:
                model.save_weights(filepath=filepath.replace('model', name),
                                   overwrite=overwrite,
                                   save_format=save_format,
                                   options=options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        """
        Load the weights for the submodels.

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

        for name in self.model_names:
            model = getattr(self, name, None)

            if model is not None:
                model.load_weights(filepath=filepath.replace('model', name),
                                   by_name=by_name,
                                   skip_mismatch=skip_mismatch,
                                   options=options)


class SynthesisBaseModel(BaseModel):
    """
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 writers_shape,
                 **kwargs):
        """
        Initialize the synthesis model with specified parameters for each submodel.

        Parameters
        ----------
        image_shape : tuple or list
            The shape of the input images.
        lexical_shape : tuple or list
            The shape of the lexical input.
        writers_shape : int
            The dimension for the writer identification.
        **kwargs : dict
            Additional keyword arguments for the TensorFlow Keras Model.
        """

        super().__init__(name='synthesis', **kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.writers_shape = writers_shape

        self.generator = None
        self.style_backbone = None
        self.style_encoder = None
        self.discriminator = None
        self.patch_discriminator = None
        self.identification = None
        self.recognition = None

        self.model_names = [
            'generator',
            'style_backbone',
            'style_encoder',
            'discriminator',
            'patch_discriminator',
            'identification',
            'recognition',
        ]

        self.monitor = 'kernel_inception_distance'
        self.build_model()

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
            'writers_shape': self.writers_shape,
        })

        return config

    def compile(self, learning_rate=0.001):
        """
        Configure the submodels for training.

        This method sets up the optimizers, loss functions, and metrics for the model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        """

        super().compile(run_eagerly=False)

        self.g_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.d_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.p_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.b_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.e_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.w_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.r_optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.l1_loss = L1Loss()
        self.ctx_loss = CTXLoss()
        self.ctc_loss = CTCLoss()
        self.kld_loss = tf.keras.losses.KLDivergence()
        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.kid = KernelInceptionDistance()

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
        self.kid.update_state(x_data[0], generated_images)

        return {
            self.kid.name: self.kid.result(),
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


class SynthesisRecognitionBaseModel(BaseModel):
    """
    A handwriting recognition base model.
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 style_backbone=None,
                 style_encoder=None,
                 generator=None,
                 synthesis_ratio=1.0,
                 **kwargs):
        """
        """

        super().__init__(name='synthesis_recognition', **kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape
        self.synthesis_ratio = synthesis_ratio

        self.recognition = None
        self.style_backbone = style_backbone
        self.style_encoder = style_encoder
        self.generator = generator

        self.model_names = [
            'recognition',
            'style_backbone',
            'style_encoder',
            'generator',
        ]

        self.monitor = 'val_edit_distance'
        self.build_model()

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
        })

        return config

    def compile(self, learning_rate=0.001):
        """
        Configure the submodels for training.

        This method sets up the optimizers, loss functions, and metrics for the model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for the optimizer.
        """

        super().compile(run_eagerly=False)

        self.optimizer = NormalizedOptimizer(
            tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.001))

        self.ctc_loss = CTCLoss()
        self.edit_distance = EditDistance()

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

        (image_inputs, text_inputs, aug_image_inputs, aug_text_inputs, _), _ = input_data

        images = aug_image_inputs
        texts = text_inputs

        if self.style_backbone and self.style_encoder and self.generator:
            if random.random() <= self.synthesis_ratio:
                images = image_inputs
                texts = aug_text_inputs

                features_inputs, _ = self.style_backbone(images, training=False)
                latent_inputs, _, _ = self.style_encoder(features_inputs, training=False)
                images = self.generator([latent_inputs, texts], training=False)

        with tf.GradientTape() as tape:
            ctc_logits = self.recognition(images, training=True)
            ctc_loss = self.ctc_loss(texts, ctc_logits)

        gradients = tape.gradient(ctc_loss, self.recognition.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.recognition.trainable_weights))

        self.edit_distance.update_state(texts, ctc_logits)

        return {
            self.ctc_loss.name: ctc_loss,
            self.edit_distance.name: self.edit_distance.result(),
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

        x_data, y_data = input_data

        ctc_logits = self.call(x_data, training=False)
        ctc_loss = self.ctc_loss(y_data, ctc_logits)

        self.edit_distance.update_state(y_data, ctc_logits)

        return {
            'ctc_loss': ctc_loss,
            'edit_distance': self.edit_distance.result(),
        }

    def call(self, x_data, training=None):
        """
        Processes input images and transcribes handwritten texts from them.

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

        image_inputs, _, _, _, _ = x_data

        ctc_logits = self.recognition(image_inputs, training=training)

        return ctc_logits
