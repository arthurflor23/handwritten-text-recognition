"""Handwritten text recognition network"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Reshape
from tensorflow.keras.layers import TimeDistributed, Activation, Dense, Bidirectional, LSTM
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from network.ctc_model import CTCModel
from contextlib import redirect_stdout
import os


class HTRNetwork:

    def __init__(self, env, dtgen):
        self.summary_path = os.path.join(env.output, "summary.txt")
        self.checkpoint_path = os.path.join(env.output, "checkpoint_weights.hdf5")
        self.logger_path = os.path.join(env.output, "logger.log")

        self._build_network(env.model_input_size, dtgen.charset, dtgen.max_text_length, dtgen.training)

        if dtgen.training:
            self._build_callbacks(env.output)

        if os.path.isfile(self.checkpoint_path):
            self.model.load_checkpoint(self.checkpoint_path)

    def summary_to_file(self):
        """Save model structure (summary) in a file"""

        with open(self.summary_path, "w") as f:
            with redirect_stdout(f):
                self.model.summary()

    def _build_network(self, input_size, charset, max_text_length, training):
        """Build the HTR network: CNN -> RNN -> CTC"""

        # build CNN
        input_data = Input(name="input", shape=(None, input_size[1], 1))

        filters = [32, 64, 128, 256]
        kernels = [5, 5, 3, 3]
        nb_layers = len(kernels)
        pool_sizes, strides = get_pool_strides(input_size, max_text_length, nb_layers)

        cnn = input_data

        for i in range(nb_layers):
            cnn = Conv2D(filters=filters[i], kernel_size=kernels[i], padding="same", activation="relu",
                         kernel_initializer="he_normal", kernel_regularizer=l2(0.01))(cnn)
            cnn = BatchNormalization(scale=False, center=True, trainable=training)(cnn)
            cnn = MaxPooling2D(pool_size=pool_sizes[i], strides=strides[i], padding="valid")(cnn)

        shape = cnn.get_shape()
        outcnn = Reshape((-1, shape[2] * shape[3]))(cnn)

        # build RNN
        blstm = Bidirectional(LSTM(units=512, return_sequences=True, kernel_initializer="he_normal"))(outcnn)
        dense = TimeDistributed(Dense(units=(len(charset) + 1), kernel_initializer="he_normal"))(blstm)
        outrnn = Activation(activation="softmax")(dense)

        # create and compile CTC model
        self.model = CTCModel(
            inputs=[input_data],
            outputs=[outrnn],
            greedy=False,
            beam_width=max_text_length,
            top_paths=1,
            charset=charset)

        self.model.compile(optimizer=Adam(learning_rate=1e-2))

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
                write_images=True,
                update_freq="epoch"
            ),
            ModelCheckpoint(
                filepath=self.checkpoint_path,
                period=1,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                min_lr=1e-3,
                factor=0.1,
                patience=5,
                verbose=1
            )
        ]


def get_pool_strides(image_size, max_text_length, nb_layers):
    """Automatic generator of the pool and strides values"""

    def get_factors(value, nb_layers):
        factores = []
        for i in range(2, value + 1):
            while value % i == 0:
                value = value / i
                factores.append(i)

        order = sorted(factores, reverse=True)
        cand = order[:nb_layers]
        order = order[nb_layers:]

        for i in range(len(cand)):
            if len(order) == 0:
                break
            cand[i] *= order.pop()

        while len(cand) < nb_layers:
            cand.append(1)
        return cand

    x = get_factors(int(image_size[0] / (max_text_length + 1)), nb_layers)
    y = get_factors(image_size[1], nb_layers)

    pool = [(w, h) for w, h in zip(x, y)]
    strides = [(w, h) for w, h in zip(x, y)]

    return pool, strides
