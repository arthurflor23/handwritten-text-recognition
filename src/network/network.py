from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, PReLU, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Activation, Dense, Bidirectional, LSTM
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adamax
from network.ctc_model import CTCModel
import tensorflow.keras.backend as K
import numpy as np
import os


class HTRNetwork:

    def __init__(self, output, dtgen):
        self.checkpoint = os.path.join(output, "checkpoint_weights.hdf5")
        self.logger = os.path.join(output, "logger.log")

        self.build_network(dtgen.nb_features, dtgen.dictionary, dtgen.padding_value, dtgen.training)
        self.build_callbacks(dtgen.training)

        if os.path.isfile(self.checkpoint):
            self.model.load_checkpoint(self.checkpoint)

    def build_network(self, nb_features, nb_labels, padding_value, training):
        """Build the HTR network: CNN -> RNN -> CTC"""

        input_data = Input(name="input", shape=(None, nb_features))

        # build CNN
        factor = int(np.sqrt(nb_features) / 2)
        full = (factor, factor)
        half = (factor / 2, factor)

        pool_sizes = [full, full, np.divide(half, 2), np.divide(half, 2)]
        strides = [half, half, np.divide(half, 2), np.divide(half, 2)]
        filters = [64, 128, 256, 512]

        cnn = K.expand_dims(input_data, axis=3)

        for i in range(4):
            cnn = Conv2D(filters=filters[i], kernel_size=5, padding="same", kernel_initializer="he_normal")(cnn)
            cnn = PReLU(shared_axes=[1, 2])(cnn)
            cnn = BatchNormalization(trainable=training)(cnn)

            cnn = Conv2D(filters=filters[i], kernel_size=3, padding="same", kernel_initializer="he_normal")(cnn)
            cnn = MaxPooling2D(pool_size=pool_sizes[i], strides=strides[i], padding="valid")(cnn)
            cnn = PReLU(shared_axes=[1, 1])(cnn)
            cnn = BatchNormalization(trainable=training)(cnn)

        outcnn = K.squeeze(cnn, axis=2)

        # build CNN
        blstm = Bidirectional(LSTM(units=512, return_sequences=True, kernel_initializer="he_normal"))(outcnn)
        dense = TimeDistributed(Dense(units=len(nb_labels) + 1, kernel_initializer="he_normal"))(blstm)
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

    def build_callbacks(self, training):
        """Build/Call callbacks to the model"""

        tensorboard = TensorBoard(
            log_dir=os.path.dirname(self.checkpoint),
            histogram_freq=1,
            profile_batch=0,
            write_graph=True,
            write_images=True,
            update_freq="epoch")

        earlystopping = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-5,
            patience=5,
            restore_best_weights=True,
            verbose=1)

        checkpoint = ModelCheckpoint(
            filepath=self.checkpoint,
            period=1,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1)

        self.callbacks = [tensorboard, earlystopping, checkpoint]
