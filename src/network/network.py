"""Handwritten text recognition network"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, PReLU, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Activation, Dense, Bidirectional, LSTM
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import backend as K
from network.ctc_model import CTCModel
from contextlib import redirect_stdout
import os


class HTRNetwork:

    def __init__(self, output, dtgen):
        os.makedirs(output, exist_ok=True)

        self.summary_path = os.path.join(output, "summary.txt")
        self.checkpoint_path = os.path.join(output, "checkpoint_weights.hdf5")
        self.logger_path = os.path.join(output, "logger.log")

        self.__build_network(dtgen.nb_layers, dtgen.nb_features, dtgen.dictionary, dtgen.training)
        self.__build_callbacks()

        if os.path.isfile(self.checkpoint_path):
            self.model.load_checkpoint(self.checkpoint_path)

    def summary_to_file(self):
        """Save model structure (summary) in a file"""

        with open(self.summary_path, "w") as f:
            with redirect_stdout(f):
                self.model.summary()

    def __build_network(self, nb_layers, nb_features, nb_labels, training):
        """Build the HTR network: CNN -> RNN -> CTC"""

        # build CNN
        input_data = Input(name="input", shape=(None, nb_features))
        filters = [64, 128, 256, 512]
        pool_sizes, strides = pool_strides(nb_features, len(filters))

        cnn = K.expand_dims(x=input_data, axis=3)

        for i in range(nb_layers):
            cnn = Conv2D(filters=filters[i], kernel_size=5, padding="same", kernel_initializer="he_normal")(cnn)
            cnn = PReLU(shared_axes=[1, 2])(cnn)
            cnn = BatchNormalization(trainable=training)(cnn)

            cnn = Conv2D(filters=filters[i], kernel_size=3, padding="same", kernel_initializer="he_normal")(cnn)
            cnn = MaxPooling2D(pool_size=pool_sizes[i], strides=strides[i], padding="valid")(cnn)
            cnn = PReLU(shared_axes=[1, 1])(cnn)
            cnn = BatchNormalization(trainable=training)(cnn)

        outcnn = K.squeeze(x=cnn, axis=2)

        # build CNN
        blstm = Bidirectional(LSTM(units=512, return_sequences=True, kernel_initializer="he_normal"))(outcnn)
        dense = TimeDistributed(Dense(units=(len(nb_labels) + 1), kernel_initializer="he_normal"))(blstm)
        outrnn = Activation(activation="softmax")(dense)

        # create and compile CTC model
        self.model = CTCModel(
            inputs=[input_data],
            outputs=[outrnn],
            greedy=False,
            beam_width=100,
            top_paths=1,
            charset=nb_labels)

        self.model.compile(optimizer=Adamax(learning_rate=0.0001))

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


def pool_strides(nb_features, nb_layers):
    """Generate pool sizes and strides values with features and layers numbers"""

    factores, pool, strides = [], [], []

    for i in range(2, nb_features + 1):
        while nb_features % i == 0:
            nb_features = nb_features / i
            factores.append(i)

    order = sorted(factores, reverse=True)
    cand = order[:nb_layers]
    order = order[nb_layers:]

    for i in range(len(cand)):
        if len(order) == 0:
            break
        cand[i] *= order.pop()

    for i in range(nb_layers):
        pool.append((int(cand[i] / 2), cand[i]))
        strides.append((1, cand[i]))

    return pool, strides
