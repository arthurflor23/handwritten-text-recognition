from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling2D, PReLU, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Activation, Dense, Bidirectional, LSTM, Masking
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adamax
from network.ctc_model import CTCModel
import tensorflow as tf
import os


class HTRNetwork:

    def __init__(self, output, dtgen):
        self.models_path = os.path.join(output, "models")
        self.checkpoint = os.path.join(output, "checkpoint_weights.hdf5")
        self.logger = os.path.join(output, "logger.log")

        self.opt = Adamax(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

        self.build_network(dtgen.nb_features, dtgen.dictionary, dtgen.padding_value, dtgen.training)
        self.build_callbacks(dtgen.training)

        os.makedirs(self.models_path, exist_ok=True)
        if os.path.isfile(os.path.join(self.models_path, "model_train.json")):
            self.model.load_model(path_dir=self.models_path, optimizer=self.opt)

    def build_network(self, nb_features, nb_labels, padding_value, training):
        """Build the HTR network: CNN -> RNN -> CTC"""

        input_data = Input(name="input", shape=(None, nb_features))

        # build CNN
        filters = [64, 128, 256, 512]
        kernels = [5, 5, 3, 3]
        pool_sizes = strides = [(2,2), (2,2), (1,1), (1,1)]

        cnn = Conv1D(filters=64, kernel_size=5, padding="same", kernel_initializer="he_normal")(input_data)
        cnn = PReLU(shared_axes=[0, 1])(cnn)
        cnn = BatchNormalization(trainable=training)(cnn)

        cnn = tf.expand_dims(input=cnn, axis=3, name="expand")

        for i in range(4):
            cnn = Conv2D(filters=filters[i], kernel_size=kernels[i], padding="same", kernel_initializer="he_normal")(cnn)
            cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)
            cnn = PReLU(shared_axes=[1, 2])(cnn)
            cnn = BatchNormalization(trainable=training)(cnn)

            cnn = Conv2D(filters=filters[i], kernel_size=kernels[i], padding="same", kernel_initializer="he_normal")(cnn)
            cnn = MaxPooling2D(pool_size=pool_sizes[i], strides=strides[i], padding="valid")(cnn)
            cnn = PReLU(shared_axes=[1, 2])(cnn)
            cnn = BatchNormalization(trainable=training)(cnn)

        outcnn = tf.squeeze(input=cnn, axis=2, name="squeeze")

        # build CNN
        masking = Masking(mask_value=padding_value)(outcnn)
        blstm = Bidirectional(LSTM(units=512, return_sequences=True, kernel_initializer="he_normal"))(masking)

        dense = TimeDistributed(Dense(units=len(nb_labels) + 1, kernel_initializer="he_normal"))(blstm)
        outrnn = Activation(activation="softmax", name="softmax")(dense)

        # create and compile
        self.model = CTCModel(
            inputs=[input_data],
            outputs=[outrnn],
            greedy=False,
            beam_width=100,
            top_paths=1,
            charset=nb_labels
        )

        self.model.compile(optimizer=self.opt)

    def build_callbacks(self, training):
        """Build/Call callbacks to the model"""

        tensorboard = TensorBoard(
            log_dir=os.path.dirname(self.models_path),
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

        self.callbacks = [tensorboard, earlystopping]
