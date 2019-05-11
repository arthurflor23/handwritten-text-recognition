"""Handwritten text recognition network"""

from tensorflow.keras.layers import Input, Conv2D, Reshape, Multiply
from tensorflow.keras.layers import Activation, Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import RMSprop
from network.ctc_model import CTCModel
from contextlib import redirect_stdout
import os


class HTRNetwork:

    def __init__(self, env, dtgen):
        self.logger_path = os.path.join(env.output, "logger.log")
        self.summary_path = os.path.join(env.output, "summary.txt")
        self.checkpoint_path = os.path.join(env.output, "checkpoint_weights.hdf5")

        self._build_network(env.model_input_size, env.charset, dtgen.max_text_length)
        self._build_callbacks(env.output)

        if os.path.isfile(self.checkpoint_path):
            self.model.load_checkpoint(self.checkpoint_path)

    def summary_to_file(self):
        """Save model structure (summary) in a file"""

        with open(self.summary_path, "w") as f:
            with redirect_stdout(f):
                self.model.summary()

    def _build_network(self, input_size, charset, max_text_length):
        """Build Gated Convolucional Recurrent Neural Network"""

        input_data = Input(name="input", shape=(None, input_size[1], 1))
        init = "glorot_uniform"

        cnn = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding='same',
                     activation="relu", kernel_initializer=init)(input_data)
        cnn = Dropout(rate=0.5)(cnn)

        cnn = Conv2D(filters=16, kernel_size=(2,4), strides=(2,4), padding='same',
                     activation="relu", kernel_initializer=init)(cnn)
        cnn = Dropout(rate=0.5)(cnn)

        cnn = GatedConv(nb_filters=16, kernel_size=(3,3))(cnn)

        cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',
                     activation="relu", kernel_initializer=init)(cnn)
        cnn = Dropout(rate=0.5)(cnn)

        cnn = GatedConv(nb_filters=32, kernel_size=(3,3))(cnn)

        cnn = Conv2D(filters=64, kernel_size=(2,4), strides=(2,4), padding='same',
                     activation="relu", kernel_initializer=init)(cnn)
        cnn = Dropout(rate=0.5)(cnn)

        cnn = GatedConv(nb_filters=64, kernel_size=(3,3))(cnn)

        cnn = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',
                     activation="relu", kernel_initializer=init)(cnn)
        cnn = Dropout(rate=0.5)(cnn)

        shape = cnn.get_shape()
        outcnn = Reshape((-1, shape[2] * shape[3]))(cnn)

        blstm = Bidirectional(LSTM(units=128, return_sequences=True, kernel_initializer=init))(outcnn)
        dense = Dense(units=128, kernel_initializer=init)(blstm)

        blstm = Bidirectional(LSTM(units=128, return_sequences=True, kernel_initializer=init))(dense)
        dense = Dense(units=128, kernel_initializer=init)(blstm)

        outrnn = Activation(activation="softmax")(dense)

        self.model = CTCModel(
            inputs=[input_data],
            outputs=[outrnn],
            greedy=False,
            beam_width=max_text_length,
            top_paths=1,
            charset=charset)

        self.model.compile(optimizer=RMSprop(learning_rate=4e-4))

    def _build_callbacks(self, output):
        """Build callbacks to the model"""

        os.makedirs(output, exist_ok=True)
        os.makedirs(os.path.join(output, "train"), exist_ok=True)
        os.makedirs(os.path.join(output, "validation"), exist_ok=True)

        self.callbacks = [
            CSVLogger(
                filename=self.logger_path,
                append=True
            ),
            TensorBoard(
                log_dir=output,
                histogram_freq=1,
                profile_batch=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch"
            ),
            ModelCheckpoint(
                filepath=self.checkpoint_path,
                period=1,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]


class GatedConv(Conv2D):
    """A Keras layer implementing gated convolutions [1]_.
    Args:
        nb_filters (int): Number of output filters.
        kernel_size (int or tuple): Size of convolution kernel.
        strides (int or tuple): Strides of the convolution.
        padding (str): One of ``'valid'`` or ``'same'``.
        kwargs: Other layer keyword arguments.
    References:
        .. [1] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier,
               “Language modeling with gated convolutional networks,” in
               Proc. 34th Int. Conf. Mach. Learn. (ICML), vol. 70,
               Sydney, Australia, 2017, pp. 933–941.
    """
    def __init__(self, nb_filters=64, kernel_size=(3, 3), **kwargs):
        super(GatedConv, self).__init__(filters=nb_filters * 2, kernel_size=kernel_size, **kwargs)
        self.nb_filters = nb_filters

    def call(self, inputs):
        """Apply gated convolution."""

        output = super(GatedConv, self).call(inputs)
        nb_filters = self.nb_filters
        linear = Activation('linear')(output[:, :, :, :nb_filters])
        sigmoid = Activation('sigmoid')(output[:, :, :, nb_filters:])

        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output."""

        output_shape = super(GatedConv, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        """Return the config of the layer."""

        config = super(GatedConv, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config
