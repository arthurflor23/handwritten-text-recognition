"""
Gated implementations
    GatedConv: Introduce a Conv2D layer (same number of filters) to multiply with its sigmoid activation.
    Gated: Introduce a Conv2D to extract features (linear and sigmoid), making a full gated process.
           This process will double number of filters to optimize convolutional process.
"""

from tensorflow.keras.layers import Conv2D, Multiply, Activation

"""
Tensorflow Keras layer implementation of the gated convolution.
    Args:
        kwargs: Conv2D keyword arguments.
    Reference:
        T. Bluche, R. Messina,
        Gated convolutional recurrent neural networks for multilingual handwriting recognition.
        14th IAPR International Conference on Document Analysis andRecognition (ICDAR),
        p. 646–651, 11 2017.
"""


class GatedConv(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, **kwargs):
        super(GatedConv, self).__init__(**kwargs)

    def call(self, inputs):
        """Apply gated convolution"""

        output = super(GatedConv, self).call(inputs)
        linear = Activation("linear")(inputs)
        sigmoid = Activation("sigmoid")(output)

        return Multiply()([linear, sigmoid])

    def get_config(self):
        """Return the config of the layer"""

        config = super(GatedConv, self).get_config()
        return config


"""
Tensorflow Keras layer implementation of the gated convolution.
    Args:
        filters (int): Number of output filters.
        kwargs: Other Conv2D keyword arguments.
    Reference:
        Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier,
        Language modeling with gated convolutional networks, in
        Proc. 34th Int. Conf. Mach. Learn. (ICML), vol. 70,
        Sydney, Australia, pp. 933–941, 2017.
"""


class Gated(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, filters, **kwargs):
        super(Gated, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        """Apply gated convolution"""

        output = super(Gated, self).call(inputs)
        linear = Activation("linear")(output[:, :, :, :self.nb_filters])
        sigmoid = Activation("sigmoid")(output[:, :, :, self.nb_filters:])

        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output"""

        output_shape = super(Gated, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        """Return the config of the layer"""

        config = super(Gated, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config
