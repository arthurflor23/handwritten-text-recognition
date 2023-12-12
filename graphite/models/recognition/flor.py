import tensorflow as tf


class HandwritingRecognition(tf.keras.Model):
    """
    """

    def __init__(self,
                 image_shape,
                 lexical_shape,
                 **kwargs):
        """
        Initialize the handwriting recognition model with specified parameters.

        Parameters
        ----------
        image_shape : list or tuple
            Shape of the input image.
        lexical_shape : list or tuple
            Shape of the text sequences and vocabulary encoding.
        **kwargs : dict
            Additional keyword arguments for `tf.keras.Model`.
        """

        super().__init__(**kwargs)

        self.image_shape = image_shape
        self.lexical_shape = lexical_shape

        self.build_model()

    def get_config(self):
        """
        Retrieves the configuration of the model for serialization.

        Returns
        -------
        dict
            A dictionary containing the configuration of the model.
        """

        config = {
            'image_shape': self.image_shape,
            'lexical_shape': self.lexical_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}

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

    def build_model(self):
        """
        Initializes and builds the neural network model.

        This method sets up the architecture of the model by defining layers, their connections,
            and configurations. It is typically called in the constructor to create the model structure.
        """

        print('build_model')

        # self.model = tf.keras.Model(inputs=image_inputs, outputs=outputs, name=self.name)

        # self.summary = self.model.summary
        # self.call = self.model.call
