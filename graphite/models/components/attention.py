import tensorflow as tf

from models.components.normalization import SpectralNormalization


class SpectralSelfAttention(tf.keras.layers.Layer):
    """
    Spectral Self-Attention layer for TensorFlow models.

    Implements a self-attention mechanism with spectral normalization, suitable for tasks like
        image generation. The layer applies self-attention to the input tensor, enhancing feature
        representations for better model performance.

    References
    ----------
    - [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
    - [Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957).
    """

    def __init__(self, **kwargs):
        """
        Initialize the SpectralSelfAttention layer.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the layer.
        """

        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the layer with spectral normalization on convolutional layers.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        """

        self.shape = input_shape
        self.num_channels = input_shape[-1]
        self.hw = input_shape[1] * input_shape[2]

        self.conv_f = SpectralNormalization(tf.keras.layers.Conv2D(self.num_channels // 8, 1))
        self.conv_g = SpectralNormalization(tf.keras.layers.Conv2D(self.num_channels // 8, 1))
        self.conv_h = SpectralNormalization(tf.keras.layers.Conv2D(self.num_channels // 2, 1))
        self.conv_o = SpectralNormalization(tf.keras.layers.Conv2D(self.num_channels, 1))

        self.gamma = self.add_weight(shape=(1,),
                                     initializer='zeros',
                                     trainable=True,
                                     name=f"{self.name}_gamma")

    def call(self, x):
        """
        Apply self-attention to the input tensor.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor to the layer.

        Returns
        -------
        tf.Tensor
            Processed tensor with self-attention applied.
        """

        f = self.conv_f(x)
        g = self.conv_g(x)
        h = self.conv_h(x)

        f = tf.reshape(tensor=f, shape=[-1, self.hw, f.shape[-1]])
        g = tf.reshape(tensor=g, shape=[-1, self.hw, g.shape[-1]])
        h = tf.reshape(tensor=h, shape=[-1, self.hw, h.shape[-1]])

        s = tf.matmul(g, f, transpose_b=True)
        beta = tf.nn.softmax(logits=s)

        o = tf.matmul(beta, h)
        o = tf.reshape(tensor=o, shape=[-1, self.shape[1], self.shape[2], self.num_channels // 2])
        o = self.conv_o(o)

        return self.gamma * o + x
