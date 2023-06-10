import tensorflow as tf


class OctConv2D(tf.keras.layers.Layer):
    """
    Octave Convolutional Layer.

    References
    ----------
    Yunpeng Chen, Haoqi Fan, Bing Xu, Zhicheng Yan, Yannis Kalantidis, Marcus Rohrbach, Shuicheng Yan, Jiashi Feng.
    Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution.
    Proceedings of the IEEE International Conference on Computer Vision, 2019.
    Github: https://github.com/koshian2/OctConv-TFKeras
    """

    def __init__(self,
                 filters,
                 alpha,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        """
        Initialize the Octave layer.

        Parameters
        ----------
        filters : int
            Number of filters for the convolutional layer.
        alpha : float
            Ratio of low-frequency filters to the total filters.
        kernel_size : tuple of ints, optional
            Size of the convolutional kernel (default is (3, 3)).
        strides : tuple of ints, optional
            Strides of the convolutional operation (default is (1, 1)).
        padding : str, optional
            Padding mode for the convolution (default is 'same').
        kernel_initializer : str or tf.keras.initializers.Initializer, optional
            Initializer for the kernel weights (default is 'glorot_uniform').
        kernel_regularizer : str or tf.keras.regularizers.Regularizer, optional
            Regularizer function applied to the kernel weights (default is None).
        kernel_constraint : str or tf.keras.constraints.Constraint, optional
            Constraint function applied to the kernel weights (default is None).
        **kwargs : dict
            Additional Layer keyword arguments.
        """

        assert alpha >= 0 and alpha <= 1
        assert filters > 0 and isinstance(filters, int)

        super().__init__(**kwargs)

        self.alpha = alpha
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        self.low_channels = int(self.filters * self.alpha)
        self.high_channels = self.filters - self.low_channels

    def build(self, input_shape):
        """
        Build the Octave layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        """

        assert len(input_shape) == 2
        assert len(input_shape[0]) == 4 and len(input_shape[1]) == 4
        assert input_shape[0][1] // 2 >= self.kernel_size[0]
        assert input_shape[0][2] // 2 >= self.kernel_size[1]
        assert input_shape[0][1] // input_shape[1][1] == 2
        assert input_shape[0][2] // input_shape[1][2] == 2
        assert tf.keras.backend.image_data_format() == 'channels_last'

        high_in = int(input_shape[0][3])
        low_in = int(input_shape[1][3])

        self.high_to_high_kernel = self.add_weight(name='high_to_high_kernel',
                                                   shape=(*self.kernel_size, high_in, self.high_channels),
                                                   initializer=self.kernel_initializer,
                                                   regularizer=self.kernel_regularizer,
                                                   constraint=self.kernel_constraint)

        self.high_to_low_kernel = self.add_weight(name='high_to_low_kernel',
                                                  shape=(*self.kernel_size, high_in, self.low_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)

        self.low_to_high_kernel = self.add_weight(name='low_to_high_kernel',
                                                  shape=(*self.kernel_size, low_in, self.high_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)

        self.low_to_low_kernel = self.add_weight(name='low_to_low_kernel',
                                                 shape=(*self.kernel_size, low_in, self.low_channels),
                                                 initializer=self.kernel_initializer,
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)

        super().build(input_shape)

    def call(self, inputs):
        """
        Apply Octave convolution to the input.

        Parameters
        ----------
        inputs : list of tensors
            List of input tensors [high_input, low_input].

        Returns
        -------
        list of tensors
            List of output tensors [high_output, low_output].
        """

        assert len(inputs) == 2

        high_input, low_input = inputs

        high_to_high = tf.keras.backend.conv2d(x=high_input,
                                               kernel=self.high_to_high_kernel,
                                               strides=self.strides,
                                               padding=self.padding,
                                               data_format='channels_last')

        high_to_low = tf.keras.backend.pool2d(x=high_input,
                                              pool_size=(2, 2),
                                              strides=(2, 2),
                                              pool_mode='avg')

        high_to_low = tf.keras.backend.conv2d(x=high_to_low,
                                              kernel=self.high_to_low_kernel,
                                              strides=self.strides,
                                              padding=self.padding,
                                              data_format='channels_last')

        low_to_high = tf.keras.backend.conv2d(x=low_input,
                                              kernel=self.low_to_high_kernel,
                                              strides=self.strides,
                                              padding=self.padding,
                                              data_format='channels_last')

        low_to_high = tf.keras.backend.repeat_elements(x=low_to_high, rep=2, axis=1)
        low_to_high = tf.keras.backend.repeat_elements(x=low_to_high, rep=2, axis=2)

        low_to_low = tf.keras.backend.conv2d(x=low_input,
                                             kernel=self.low_to_low_kernel,
                                             strides=self.strides,
                                             padding=self.padding,
                                             data_format='channels_last')

        high_add = high_to_high + low_to_high
        low_add = low_to_low + high_to_low

        return [high_add, low_add]

    def compute_output_shape(self, input_shapes):
        """
        Compute the output shapes of the Octave layer.

        Parameters
        ----------
        input_shapes : list of tuples
            List of input shape tuples [(high_in_shape), (low_in_shape)].

        Returns
        -------
        list of tuples
            List of output shape tuples [(high_out_shape), (low_out_shape)].
        """

        high_in_shape, low_in_shape = input_shapes
        high_out_shape = (*high_in_shape[:3], self.high_channels)
        low_out_shape = (*low_in_shape[:3], self.low_channels)

        return [high_out_shape, low_out_shape]

    def get_config(self):
        """
        Get the configuration of the Octave layer.

        Returns
        -------
        dict
            Configuration dictionary of the layer.
        """

        config = super().get_config()

        config = {
            **config,
            'filters': self.filters,
            'alpha': self.alpha,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
        }

        return config
