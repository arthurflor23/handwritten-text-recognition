"""Create the HTR deep learning model using tf.keras"""

import os
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, BatchNormalization, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Lambda
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class HTR():

    def __init__(self, args, training=False):
        self.dictionary = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.input_shape = (800, 64, 1)
        self.max_line_length = 140

        self.batch_size = args.batch
        self.training = training

        self.output = args.output
        self.checkpoint = os.path.join(self.output, "checkpoint_weights.hdf5")
        self.logger = os.path.join(self.output, "logger.log")

        self.__build_model()

    def __build_model(self):
        """Build all model structure"""

        input_data = Input(name="the_inputs", shape=self.input_shape, dtype="float32")
        labels = Input(name="the_labels", shape=[self.max_line_length], dtype="float32")

        input_length = Input(name="input_length", shape=[1], dtype="int32")
        label_length = Input(name="label_length", shape=[1], dtype="int32")

        cnn_out = self.__setup_cnn(input_data)
        rnn_out = self.__setup_rnn(cnn_out)

        loss_args = [rnn_out, labels, input_length, label_length]
        ctc_loss = Lambda(ctc_loss_func, output_shape=(1,), name="ctc_loss")(loss_args)

        input_args = [input_data, labels, input_length, label_length]

        self.model = Model(name="HTR", inputs=input_args, outputs=ctc_loss)
        self.model.compile(loss={"ctc_loss": lambda y_true, y_pred: y_pred}, optimizer="adamax")

        if os.path.isfile(self.checkpoint):
            self.model.load_weights(filepath=self.checkpoint, by_name=True)

    def __setup_cnn(self, input_data):
        """Build CNN"""

        kernels = [5, 5, 3, 3, 3]
        pool_sizes = strides = [(2,2), (2,2), (1,2), (1,2), (1,2)]

        self.downsample_factor = sum([p[0] for p in pool_sizes if p[0] > 1])
        self.filters = [64, 128, 256, 256, 512]

        cnn = input_data
        init = tf.random_normal_initializer(stddev=0.1)

        for i in range(len(strides)):
            conv = Conv2D(filters=self.filters[i], kernel_size=kernels[i], padding="same",
                          activation="relu", kernel_initializer=init)(cnn)
            norm = BatchNormalization(trainable=self.training)(conv)
            cnn = MaxPooling2D(pool_size=pool_sizes[i], strides=strides[i], padding="valid")(norm)

        return cnn

    def __setup_rnn(self, cnn_out):
        """Build RNN"""

        conv_to_rnn_dims = (self.input_shape[:2][0] // self.downsample_factor,
                            (self.input_shape[:2][1] // self.downsample_factor) * self.filters[0])
        inner = Reshape(target_shape=conv_to_rnn_dims)(cnn_out)

        num_units = self.filters[-1]
        lstm = LSTM(units=num_units, return_sequences=True, kernel_initializer="he_normal")
        bilstm = Bidirectional(layer=lstm)(inner)

        bilstm = Dense(units=(len(self.dictionary) + 1), kernel_initializer="he_normal")(bilstm)
        bilstm = Activation(name="softmax", activation="softmax")(bilstm)

        return bilstm

    def get_callbacks(self):
        """Build/Call callbacks to the model"""

        os.makedirs(self.output, exist_ok=True)
        logger = CSVLogger(filename=self.logger, append=True)

        tensorboard = TensorBoard(
            log_dir=self.output,
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
            save_weights_only=False,
            verbose=1)

        return [logger, tensorboard, earlystopping, checkpoint]


def ctc_loss_func(args):
    """Calculate the CTC loss function"""

    y_pred, y_true, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return ctc_batch_cost(y_true, y_pred, input_length, label_length)
