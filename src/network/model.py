"""Handwritten Text Recognition Neural Network"""

import os
import numpy as np

from contextlib import redirect_stdout
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model

from network.layers import FullGatedConv2D, GatedConv2D
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, MaxPooling2D, Reshape, TimeDistributed
from tensorflow.keras.layers import Lambda, Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import Progbar


"""
HTRModel Class based on:
    Y. Soullard, C. Ruffino and T. Paquet,
    CTCModel: A Connectionnist Temporal Classification implementation for Keras.
    ee: https://arxiv.org/abs/1901.07957, 2019.
    github: https://github.com/ysoullard/HTRModel


The HTRModel class use Tensorflow 2 Keras module for the use of the
Connectionist Temporal Classification (CTC) with the Hadwritten Text Recognition (HTR).

The HTRModel structure is composed of 2 branches. Each branch is a Tensorflow Keras Model:
    - One for computing the CTC loss (model)
    - One for predicting using the ctc_decode method (model_infer) or just returning the raw data.

In a Tensorflow Keras Model, x is the input features and y the labels.
Here, x data are of the form [input_sequences, label_sequences, inputs_lengths, labels_length]
and y are not used as in a Tensorflow Keras Model (this is an array which is not considered,
the labeling is given in the x data structure).
"""


class HTRModel:

    def __init__(self,
                 architecture,
                 input_size,
                 vocab_size,
                 greedy=False,
                 beam_width=100,
                 top_paths=1):
        """
        Initialization of a HTR Model.

        :param
            architecture: option of the architecture model to build and compile
            greedy, beam_width, top_paths: Parameters of the CTC decoding (see ctc decoding tensorflow for more details)
        """

        self.architecture = globals()[architecture]
        self.input_size = input_size
        self.vocab_size = vocab_size

        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = max(1, top_paths)

        self.model = None
        self.model_infer = None

    def summary(self, output=None, target=None):
        """Show/Save model structure (summary)"""

        self.model.summary()

        if target is not None:
            os.makedirs(output, exist_ok=True)

            with open(os.path.join(output, target), "w") as f:
                with redirect_stdout(f):
                    self.model.summary()

    def load_checkpoint(self, target):
        """ Load a model with checkpoint file"""

        if os.path.isfile(target):
            if self.model is None:
                self.compile()

            self.model.load_weights(target)
            self.model_infer.load_weights(target)

    def get_callbacks(self, logdir, checkpoint, monitor="val_loss", verbose=0):
        """Setup the list of callbacks for the model"""

        callbacks = [
            CSVLogger(
                filename=os.path.join(logdir, "epochs.log"),
                separator=";",
                append=True),
            TensorBoard(
                log_dir=logdir,
                histogram_freq=10,
                profile_batch=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch"),
            ModelCheckpoint(
                filepath=checkpoint,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose),
            EarlyStopping(
                monitor=monitor,
                min_delta=1e-8,
                patience=40,
                restore_best_weights=True,
                verbose=verbose),
            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=1e-8,
                factor=0.2,
                patience=20,
                verbose=verbose)
        ]

        return callbacks

    def compile(self, learning_rate=None):
        """
        Configures the HTR Model for training/predict.

        There are 2 Tensorflow Keras models:
            - one for training
            - one for predicting (with/without CTC decode)

        Lambda layers are used to compute:
            - the CTC loss function
            - the CTC decoding

        :param optimizer: The optimizer used during training
        """

        # define inputs, outputs and optimizer of the chosen architecture
        outs = self.architecture(self.input_size, self.vocab_size + 1, learning_rate)
        inputs, outputs, optimizer = outs

        # others inputs for the CTC approach
        labels = Input(name="labels", shape=[None])
        input_length = Input(name="input_length", shape=[1])
        label_length = Input(name="label_length", shape=[1])

        # lambda layer for computing the loss function
        loss_out = Lambda(self.ctc_loss_lambda_func, output_shape=(1,),
                          name="CTCloss")([outputs, labels, input_length, label_length])

        # lambda layer for the raw data function
        out_raw_dense = Lambda(lambda y_pred: y_pred[0], output_shape=(None, None), name="NoCTCdecode",
                               dtype="float32")([outputs, input_length])

        # create Tensorflow Keras models
        self.model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
        self.model_infer = Model(inputs=[inputs, input_length], outputs=out_raw_dense)

        # compile models
        self.model.compile(loss={"CTCloss": lambda yt, yp: yp}, optimizer=optimizer)
        self.model_infer.compile(loss={"NoCTCdecode": lambda yt, yp: yp}, optimizer=optimizer)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs):
        """
        Model training on data yielded (fit function has support to generator).
        A fit() abstration function of TensorFlow 2 using the model_train.

        Provide x parameter of the form: (x, y, sample_weight), where:
            x:  inputs = {
                    "input": x_valid,
                    "labels": y_valid,
                    "input_length": x_valid_len,
                    "label_length": y_valid_len
                }
            y:  output = {
                    "CTCloss": np.zeros(self.batch_size, dtype=int)
                }
            sample_weight: []

        yielding: (inputs, output, [])

        :param: See tensorflow.keras.Model.fit()
        :return: A history object
        """

        out = self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                             callbacks=callbacks, validation_split=validation_split,
                             validation_data=validation_data, shuffle=shuffle,
                             class_weight=class_weight, sample_weight=sample_weight,
                             initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                             validation_steps=validation_steps, validation_freq=validation_freq,
                             max_queue_size=max_queue_size, workers=workers,
                             use_multiprocessing=use_multiprocessing, **kwargs)
        return out

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=1,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                ctc_decode=True):
        """
        Model predicting on data yielded (predict function has support to generator).
        A predict() abstration function of TensorFlow 2 using the model_raw_pred or model_pred.

        Provide x parameter of the form: [x_test, x_test_len]

        :param: See tensorflow.keras.Model.predict()
        :return: raw data on `ctc_decode=False` or CTC decode on `ctc_decode=True` (both with probabilities)
        """

        self.model_infer._make_predict_function()

        if verbose == 1:
            print("Model Predict")

        out = self.model_infer.predict(x=x, batch_size=batch_size, verbose=verbose,
                                       steps=steps, callbacks=callbacks,
                                       max_queue_size=max_queue_size, workers=workers,
                                       use_multiprocessing=use_multiprocessing)

        if not ctc_decode:
            return out

        batch_size = len(out)
        max_text_length = len(max(out, key=len))

        if verbose == 1:
            print("CTC Decode")

            steps_done = 0
            progbar = Progbar(target=batch_size)

        predicts, probabilities = [], []

        for i in range(batch_size):
            decode, log = K.ctc_decode(np.asarray([out[i]]),
                                       np.asarray([max_text_length]),
                                       greedy=self.greedy,
                                       beam_width=self.beam_width,
                                       top_paths=self.top_paths)

            probabilities.extend(log)
            predicts.append(decode)

            if verbose == 1:
                steps_done += 1
                progbar.update(steps_done)

        print(probabilities)

        probabilities = [np.exp(x) for x in probabilities]
        predicts = [[[int(p) for p in x[0] if p != -1] for x in y] for y in predicts]

        return (predicts, probabilities)

    @staticmethod
    def ctc_loss_lambda_func(args):
        """
        Function for computing the ctc loss (can be put in a Lambda layer)
        :param args:
            y_pred, labels, input_length, label_length
        :return: CTC loss
        """

        y_pred, labels, input_length, label_length = args

        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


"""
Networks to the Handwritten Text Recognition Model

Reference:
    Moysset, B. and Messina, R.:
    Are 2D-LSTM really dead for offline text recognition?
    In: International Journal on Document Analysis and Recognition (IJDAR)
    Springer Science and Business Media LLC
    URL: http://dx.doi.org/10.1007/s10032-019-00325-0
"""


def bluche(input_size, output_size, learning_rate):
    """
    Gated Convolucional Recurrent Neural Network by Bluche et al.

    Reference:
        Bluche, T., Messina, R.:
        Gated convolutional recurrent neural networks for multilingual handwriting recognition.
        In: Document Analysis and Recognition (ICDAR), 2017
        14th IAPR International Conference on, vol. 1, pp. 646–651, 2017.
        URL: https://ieeexplore.ieee.org/document/8270042
    """

    input_data = Input(name="input", shape=input_size)
    cnn = Reshape((input_size[0] // 2, input_size[1] // 2, input_size[2] * 4))(input_data)

    cnn = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh")(cnn)

    cnn = Conv2D(filters=16, kernel_size=(2,4), strides=(2,4), padding="same", activation="tanh")(cnn)
    cnn = GatedConv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh")(cnn)
    cnn = GatedConv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)

    cnn = Conv2D(filters=64, kernel_size=(2,4), strides=(2,4), padding="same", activation="tanh")(cnn)
    cnn = GatedConv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh")(cnn)
    cnn = MaxPooling2D(pool_size=(1,4), strides=(1,4), padding="valid")(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    blstm = Dense(units=128, activation="tanh")(blstm)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    output_data = Dense(units=output_size, activation="softmax")(blstm)

    if learning_rate is None:
        learning_rate = 4e-4

    optimizer = RMSprop(learning_rate=learning_rate)

    return (input_data, output_data, optimizer)


def puigcerver(input_size, output_size, learning_rate):
    """
    Convolucional Recurrent Neural Network by Puigcerver et al.

    Reference:
        Puigcerver, J.: Are multidimensional recurrent layers really
        necessary for handwritten text recognition? In: Document
        Analysis and Recognition (ICDAR), 2017 14th
        IAPR International Conference on, vol. 1, pp. 67–72. IEEE (2017)
    """

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same")(input_data)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=80, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    output_data = Dense(units=output_size, activation="softmax")(blstm)

    if learning_rate is None:
        learning_rate = 3e-4

    optimizer = RMSprop(learning_rate=learning_rate)

    return (input_data, output_data, optimizer)


def flor(input_size, output_size, learning_rate):
    """
    Gated Convolucional Recurrent Neural Network by Flor et al.
    """

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding="same")(input_data)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = FullGatedConv2D(filters=16, kernel_size=(3,3), padding="same")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = FullGatedConv2D(filters=32, kernel_size=(3,3), padding="same")(cnn)

    cnn = Conv2D(filters=40, kernel_size=(2,4), strides=(2,2), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = FullGatedConv2D(filters=40, kernel_size=(3,3), padding="same", kernel_constraint=MaxNorm(4, [0,1,2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=48, kernel_size=(3,3), strides=(1,2), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = FullGatedConv2D(filters=48, kernel_size=(3,3), padding="same", kernel_constraint=MaxNorm(4, [0,1,2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=56, kernel_size=(2,4), strides=(1,2), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = FullGatedConv2D(filters=56, kernel_size=(3,3), padding="same", kernel_constraint=MaxNorm(4, [0,1,2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(cnn)
    cnn = PReLU(shared_axes=[1,2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = TimeDistributed(Dense(units=128))(bgru)

    bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = TimeDistributed(Dense(units=output_size, activation="softmax"))(bgru)

    if learning_rate is None:
        learning_rate = 5e-4

    optimizer = RMSprop(learning_rate=learning_rate)

    return (input_data, output_data, optimizer)
