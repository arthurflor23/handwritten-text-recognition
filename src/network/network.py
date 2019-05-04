"""Handwritten text recognition network"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, PReLU
from tensorflow.keras.layers import TimeDistributed, Activation, Dense, Bidirectional, LSTM
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import backend as k
from network.ctc_model import CTCModel
from contextlib import redirect_stdout
import os


class HTRNetwork:

    def __init__(self, env, dtgen):
        os.makedirs(env.output, exist_ok=True)

        self.summary_path = os.path.join(env.output, "summary.txt")
        self.checkpoint_path = os.path.join(env.output, "checkpoint_weights.hdf5")
        self.logger_path = os.path.join(env.output, "logger.log")

        self.__build_network(env.input_img_size, dtgen.dictionary, dtgen.training)
        self.__build_callbacks()

        if os.path.isfile(self.checkpoint_path):
            self.model.load_checkpoint(self.checkpoint_path)

    def summary_to_file(self):
        """Save model structure (summary) in a file"""

        with open(self.summary_path, "w") as f:
            with redirect_stdout(f):
                self.model.summary()

    def __build_network(self, img_size, dictionary, training):
        """Build the HTR network: CNN -> RNN -> CTC"""

        # build CNN
        input_data = Input(name="input", shape=(None, img_size[1]))

        filters = [32, 64, 128, 128, 256]
        kernels = [5, 5, 3, 3, 3]
        pool_sizes = strides = [(2,4), (2,2), (1,2), (1,2), (1,2)]
        nb_layers = len(strides)

        cnn = k.expand_dims(input_data, axis=3)

        for i in range(nb_layers):
            # activation="relu" (?)
            cnn = Conv2D(filters=filters[i], kernel_size=kernels[i], padding="same", kernel_initializer="he_normal")(cnn)
            cnn = BatchNormalization(trainable=training)(cnn)
            cnn = PReLU(shared_axes=[1,2])(cnn)
            cnn = MaxPooling2D(pool_size=pool_sizes[i], strides=strides[i], padding="valid")(cnn)

        outcnn = k.squeeze(cnn, axis=2)

        # build RNN
        blstm = Bidirectional(LSTM(units=256, return_sequences=True, kernel_initializer="he_normal"))(outcnn)
        dense = TimeDistributed(Dense(units=(len(dictionary) + 1), kernel_initializer="he_normal"))(blstm)
        outrnn = Activation(activation="softmax")(dense)

        # create and compile CTC model
        self.model = CTCModel(
            inputs=[input_data],
            outputs=[outrnn],
            greedy=False,
            beam_width=100,
            top_paths=1,
            charset=dictionary)

        self.model.compile(optimizer=Adamax(learning_rate=0.001))

    def __build_callbacks(self):
        """Build/Call callbacks to the model"""

        self.callbacks = [
            CSVLogger(
                filename=self.logger_path,
                append=True
            ),
            TensorBoard(
                log_dir=os.path.dirname(self.logger_path),
                histogram_freq=1,
                profile_batch=0,
                write_graph=True,
                write_images=True,
                update_freq="epoch"
            ),
            EarlyStopping(
                monitor="val_loss",
                min_delta=1e-5,
                patience=6,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.checkpoint_path,
                period=1,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]
