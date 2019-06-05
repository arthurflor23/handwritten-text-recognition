"""Handwritten Text Recognition Neural Network"""

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Lambda, TimeDistributed, Activation
from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer, Progbar
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import backend as K
from contextlib import redirect_stdout
import tensorflow as tf
import editdistance
import numpy as np
import pickle
import os


"""
HTRModel class:
    Reference:
        Y. Soullard, C. Ruffino and T. Paquet,
        CTCModel: A Connectionnist Temporal Classification implementation for Keras.
        ee: https://arxiv.org/abs/1901.07957, 2019.
        github: https://github.com/ysoullard/HTRModel


The HTRModel class extends the Tensorflow Keras Model (version 2)
for the use of the Connectionist Temporal Classification (CTC) with the Hadwritten Text Recognition (HTR).
One makes use of the CTC proposed in tensorflow. Thus HTRModel can only be used with the backend tensorflow.

The HTRModel structure is composed of 2 branches. Each branch is a Tensorflow Keras Model:
    - One for computing the CTC loss (model_train)
    - One for predicting using the ctc_decode method (model_pred) and
        computing the Character Error Rate (CER), Word Error Rate (WER), Sequence Error Rate (SER).

In a Tensorflow Keras Model, x is the input features and y the labels.
Here, x data are of the form [input_sequences, label_sequences, inputs_lengths, labels_length]
and y are not used as in a Tensorflow Keras Model (this is an array which is not considered,
the labeling is given in the x data structure).
"""


class HTRModel:

    def __init__(self, inputs, outputs, charset, greedy=False, beam_width=100, top_paths=1):
        """
        Initialization of a HTR Model.
        :param inputs: Input layer of the neural network
            outputs: Last layer of the neural network before CTC (e.g. a TimeDistributed Dense)
            greedy, beam_width, top_paths: Parameters of the CTC decoding (see ctc decoding tensorflow for more details)
            charset: labels related to the input of the CTC approach
        """
        self.model_train = None
        self.model_pred = None

        if not isinstance(inputs, list):
            self.inputs = [inputs]
        else:
            self.inputs = inputs
        if not isinstance(outputs, list):
            self.outputs = [outputs]
        else:
            self.outputs = outputs

        self.charset = charset
        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = top_paths

    def compile(self, optimizer):
        """
        Configures the HTR Model for training.

        There is 2 Tensorflow Keras models:
            - one for training
            - one for predicting/evaluate

        Lambda layers are used to compute:
            - the CTC loss function
            - the CTC decoding

        :param optimizer: The optimizer used during training
        """

        # Others inputs for the CTC approach
        labels = Input(name="labels", shape=[None])
        input_length = Input(name="input_length", shape=[1])
        label_length = Input(name="label_length", shape=[1])

        # Lambda layer for computing the loss function
        loss_out = Lambda(self.ctc_loss_lambda_func, output_shape=(1,), name="CTCloss")(
            self.outputs + [labels, input_length, label_length])

        # Lambda layer for the decoding function
        out_decoded_dense = Lambda(self.ctc_complete_decoding_lambda_func, output_shape=(None, None), name="CTCdecode",
                                   arguments={"greedy": self.greedy, "beam_width": self.beam_width, "top_paths": self.top_paths},
                                   dtype="float32")(self.outputs + [input_length])

        # create Tensorflow Keras models
        self.model_init = Model(inputs=self.inputs, outputs=self.outputs)
        self.model_train = Model(inputs=self.inputs + [labels, input_length, label_length], outputs=loss_out)
        self.model_pred = Model(inputs=self.inputs + [input_length], outputs=out_decoded_dense)

        # Compile models
        self.model_train.compile(loss={"CTCloss": lambda yt, yp: yp}, optimizer=optimizer)
        self.model_pred.compile(loss={"CTCdecode": lambda yt, yp: yp}, optimizer=optimizer)

    def get_loss_generator(self, generator, nb_batchs, verbose=False):
        """
        The generator must provide x as [input_sequences, label_sequences, inputs_lengths, labels_length]
        :return: loss on the entire dataset_manager and the loss per data
        """

        loss_per_data = []

        for k in range(nb_batchs):

            data = next(generator)
            x = data[0]
            x_len = data[2]
            y = data[1]
            y_len = data[3]
            batch_size = x.shape[0]

            no_lab = True if 0 in y_len else False

            if no_lab is False:
                loss_data = self.model_train.predict([x, y, x_len, y_len], batch_size=batch_size, verbose=verbose)
                loss_per_data += [elmt[0] for elmt in loss_data]

        return np.sum(loss_per_data), loss_per_data

    def fit_generator(self, generator,
                      steps_per_epoch,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      shuffle=True,
                      initial_epoch=0):
        """
        Model training on data yielded batch-by-batch by a Python generator.

        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.

        A major modification concerns the generator that must provide x data of the form:
          [input_sequences, label_sequences, inputs_lengths, labels_length]
        (in a similar way than for using CTC in tensorflow)

        :param: See tensorflow.keras.engine.Model.fit_generator()
        :return: A History object
        """
        out = self.model_train.fit_generator(generator, steps_per_epoch, epochs=epochs, verbose=verbose,
                                             callbacks=callbacks, validation_data=validation_data,
                                             validation_steps=validation_steps, class_weight=class_weight,
                                             max_queue_size=max_queue_size, workers=workers, shuffle=True,
                                             initial_epoch=initial_epoch)

        self.model_pred.set_weights(self.model_train.get_weights())
        return out

    def predict_generator(self,
                          generator,
                          steps,
                          max_queue_size=10,
                          workers=1,
                          verbose=0,
                          decode_func=None):
        """Generates predictions and evaluations (loss, cer, wer, ser)
        for the input samples from a data generator.

        The generator should return the same kind of data as accepted by `predict_on_batch`.

        generator = DataGenerator class that returns:
                        x = Input data as a 3D Tensor (batch_size, max_input_len, dim_features)
                        x_len = 1D array with the length of each data in batch_size
                        y = Input data (bytes format) as a 2D Tensor (batch_size, max_label_len)

        # Arguments
            generator: Generator yielding batches of input samples
                    or an instance of Sequence (tensorflow.keras.utils.Sequence)
                    object in order to avoid duplicate data
                    when using multiprocessing.
            steps:
                Total number of steps (batches of samples)
                to yield from `generator` before stopping.
            max_queue_size:
                Maximum size for the generator queue.
            workers: Maximum number of processes to spin up
                when using process based threading
            verbose:
                verbosity mode, 0 or 1.
            decode_func:
                a function for decoding a list of predicted sequences (using self.charset)

        # Returns
            A tuple containing:
                A numpy array(s) of ground truth.
                A numpy array(s) of predictions.
            A list containing the error rate:
                cer : the character error rate on the dataset
                wer : the word error rate on the dataset
                ser : the sequence error rate on the dataset

        # Raises
            ValueError: In case the generator yields
                data in an invalid format.
        """

        self.model_pred._make_predict_function()
        is_sequence = isinstance(generator, Sequence)

        allab_outs, all_lab = [], []
        steps_done = 0
        enqueuer = None

        try:
            if is_sequence:
                enqueuer = OrderedEnqueuer(generator)
            else:
                enqueuer = GeneratorEnqueuer(generator)

            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()

            if verbose == 1:
                progbar = Progbar(target=steps)

            while steps_done < steps:
                generator_output = next(output_generator)

                if isinstance(generator_output, tuple):
                    if len(generator_output) == 2:
                        x, _ = generator_output
                    elif len(generator_output) == 3:
                        x, _, _ = generator_output
                    else:
                        raise ValueError("Output of generator should be a tuple `(x, y, sample_weight)` "
                                         "or `(x, y)`. Found: " + str(generator_output))
                else:
                    x = generator_output

                [x_input, x_length, y] = x

                predict = self.model_pred.predict_on_batch([x_input, x_length])
                outs = [[pr for pr in pred if pr != -1] for pred in predict]

                if not isinstance(outs, list):
                    outs = [outs]

                if not allab_outs:
                    for out in outs:
                        allab_outs.append([])
                        all_lab.append([])

                for i, out in enumerate(outs):
                    all_lab[i].append(y[i])
                    allab_outs[i].append([valab_out for valab_out in out if valab_out != -1])

                steps_done += 1
                if verbose == 1:
                    progbar.update(steps_done)

        finally:
            if enqueuer is not None:
                enqueuer.stop()

        batch_size = len(allab_outs)
        nb_data = len(allab_outs[0])

        lab_out, pred_out = [], []
        cer, wer, ser = [], [], []

        for i in range(nb_data):
            lab_out += [all_lab[b][i] for b in range(batch_size)]
            pred_out += [allab_outs[b][i] for b in range(batch_size)]

        # decode
        lab_out = np.array([b.decode() for b in lab_out])
        pred_out = np.array([("".join(self.charset[int(c)] for c in l)).strip() for l in pred_out])

        # error rate calculations
        for (lab, pred) in zip(lab_out, pred_out):
            pd, lb = list(pred), list(lab)
            length = max(len(pd), len(lb))
            dist = editdistance.eval(pd, lb)
            cer.append(dist / length)

            pd, lb = pred.split(), lab.split()
            length = max(len(pd), len(lb))
            dist = editdistance.eval(pd, lb)
            wer.append(dist / length)

            pd, lb = [pred], [lab]
            length = max(len(pd), len(lb))
            dist = editdistance.eval(pd, lb)
            ser.append(dist / length)

        cer = sum(cer) / len(cer)
        wer = sum(wer) / len(wer)
        ser = sum(ser) / len(ser)

        return [lab_out, pred_out], [cer, wer, ser]

    def save_model(self, path_dir, charset=None):
        """ Save a model in path_dir
        save model_train, model_pred in json
        save inputs and outputs in json
        save model CTC parameters in a pickle

        :param path_dir: directory where the model architecture will be saved
        :param charset: set of labels (useful to keep the label order)
        """

        model_json = self.model_train.to_json()
        with open(path_dir + "/model_train.json", "w") as json_file:
            json_file.write(model_json)

        model_json = self.model_pred.to_json()
        with open(path_dir + "/model_pred.json", "w") as json_file:
            json_file.write(model_json)

        model_json = self.model_init.to_json()
        with open(path_dir + "/model_init.json", "w") as json_file:
            json_file.write(model_json)

        param = {"greedy": self.greedy, "beam_width": self.beam_width, "top_paths": self.top_paths, "charset": self.charset}

        output = open(path_dir + "/model_param.pkl", "wb")
        p = pickle.Pickler(output)
        p.dump(param)
        output.close()

    def load_checkpoint(self, checkpoint):
        """ Load a model with checkpoint file
        load model_train, model_pred from hdf5
        """

        if os.path.isfile(checkpoint):
            self.model_train.load_weights(checkpoint)
            self.model_pred.set_weights(self.model_train.get_weights())

    def load_model(self, path_dir, optimizer, init_archi=True, file_weights=None, change_parameters=False,
                   init_last_layer=False, add_layers=None, trainable=False, removed_layers=2):
        """ Load a model in path_dir
        load model_train, model_pred from json
        load inputs and outputs from json
        load model CTC parameters from a pickle

        :param path_dir: directory where the model is saved
        :param optimizer: The optimizer used during training
        :param init_archi: load an architecture from json. Otherwise, the network archtiecture muste be initialized.
        :param file_weights: weights to load (None = default parameters are returned).
        :param init_last_layer: reinitialize the last layer using self.charset to get the number of labels.
        :param add_layers: add some layers. None for no change in the network architecture. Otherwise, add_layers contains
        a list of layers to add after the last layer of the current architecture.
        :param trainable: in case of add_layers, lower layers can be not trained again.
        :param removed_layers: remove the last layers of the current architecture. It is applied before adding new layers using add_layers.
        """

        if init_archi:
            json_file = open(path_dir + "/model_train.json", "r")
            loaded_model_json = json_file.read()
            json_file.close()
            self.model_train = model_from_json(loaded_model_json)

            json_file = open(path_dir + "/model_pred.json", "r")
            loaded_model_json = json_file.read()
            json_file.close()
            self.model_pred = model_from_json(loaded_model_json, custom_objects={"tf": tf})

            json_file = open(path_dir + "/model_init.json", "r")
            loaded_model_json = json_file.read()
            json_file.close()
            self.model_init = model_from_json(loaded_model_json, custom_objects={"tf": tf})

            self.inputs = self.model_init.inputs
            self.outputs = self.model_init.outputs

            input = open(path_dir + "/model_param.pkl", "rb")
            p = pickle.Unpickler(input)
            param = p.load()
            input.close()

            if not change_parameters:
                self.greedy = param["greedy"] if "greedy" in param.keys() else self.greedy
                self.beam_width = param["beam_width"] if "beam_width" in param.keys() else self.beam_width
                self.top_paths = param["top_paths"] if "top_paths" in param.keys() else self.top_paths
            self.charset = param["charset"] if "charset" in param.keys() and self.charset is None else self.charset

        self.compile(optimizer)

        if file_weights is not None:
            if os.path.exists(file_weights):
                self.model_train.load_weights(file_weights)
                self.model_pred.set_weights(self.model_train.get_weights())
            elif os.path.exists(path_dir + file_weights):
                self.model_train.load_weights(path_dir + file_weights)
                self.model_pred.set_weights(self.model_train.get_weights())

        # add layers after transfer
        if add_layers is not None:
            labels = Input(name="labels", shape=[None])
            input_length = Input(name="input_length", shape=[1])
            label_length = Input(name="label_length", shape=[1])

            new_layer = Input(name="input", shape=self.model_init.layers[0].output_shape[1:])
            self.inputs = [new_layer]
            for layer in self.model_init.layers[1:-removed_layers]:
                print(layer)
                new_layer = layer(new_layer)
                layer.trainable = trainable
            for layer in add_layers:
                new_layer = layer(new_layer)
                layer.trainable = True

            self.outputs = [new_layer]
            loss_out = Lambda(self.ctc_loss_lambda_func, output_shape=(1,), name="CTCloss")(
                self.outputs + [labels, input_length, label_length])

            # Lambda layer for the decoding function
            out_decoded_dense = Lambda(self.ctc_complete_decoding_lambda_func, output_shape=(None, None),
                                       name="CTCdecode", arguments={"greedy": self.greedy,
                                                                    "beam_width": self.beam_width,
                                                                    "top_paths": self.top_paths}, dtype="float32")(
                self.outputs + [input_length])

            # create Tensorflow Keras models
            self.model_init = Model(inputs=self.inputs, outputs=self.outputs)
            self.model_train = Model(inputs=self.inputs + [labels, input_length, label_length], outputs=loss_out)
            self.model_pred = Model(inputs=self.inputs + [input_length], outputs=out_decoded_dense)

            # Compile models
            self.model_train.compile(loss={"CTCloss": lambda yt, yp: yp}, optimizer=optimizer)
            self.model_pred.compile(loss={"CTCdecode": lambda yt, yp: yp}, optimizer=optimizer)

        elif init_last_layer:
            labels = Input(name="labels", shape=[None])
            input_length = Input(name="input_length", shape=[1])
            label_length = Input(name="label_length", shape=[1])

            new_layer = Input(name="input", shape=self.model_init.layers[0].output_shape[1:])
            self.inputs = [new_layer]
            for layer in self.model_init.layers[1:-2]:
                new_layer = layer(new_layer)
            new_layer = TimeDistributed(Dense(len(self.charset) + 1), name="DenseSoftmax")(new_layer)
            new_layer = Activation("softmax", name="Softmax")(new_layer)

            self.outputs = [new_layer]

            # Lambda layer for computing the loss function
            loss_out = Lambda(self.ctc_loss_lambda_func, output_shape=(1,), name="CTCloss")(
                self.outputs + [labels, input_length, label_length])

            # Lambda layer for the decoding function
            out_decoded_dense = Lambda(self.ctc_complete_decoding_lambda_func, output_shape=(None, None),
                                       name="CTCdecode", arguments={"greedy": self.greedy,
                                                                    "beam_width": self.beam_width,
                                                                    "top_paths": self.top_paths}, dtype="float32")(
                self.outputs + [input_length])

            # create Tensorflow Keras models
            self.model_init = Model(inputs=self.inputs, outputs=self.outputs)
            self.model_train = Model(inputs=self.inputs + [labels, input_length, label_length], outputs=loss_out)
            self.model_pred = Model(inputs=self.inputs + [input_length], outputs=out_decoded_dense)

            # Compile models
            self.model_train.compile(loss={"CTCloss": lambda yt, yp: yp}, optimizer=optimizer)
            self.model_pred.compile(loss={"CTCdecode": lambda yt, yp: yp}, optimizer=optimizer)

    def set_callbacks(self, output):
        os.makedirs(os.path.join(output), exist_ok=True)
        checkpoint_weights = os.path.join(output, "checkpoint_weights.hdf5")

        self.callbacks = [
            TensorBoard(
                log_dir=output,
                histogram_freq=1,
                profile_batch=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch"
            ),
            ModelCheckpoint(
                filepath=checkpoint_weights,
                period=1,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.0001,
                patience=80,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        return checkpoint_weights

    def summary(self, output=None):
        """Show/Save model structure (summary)"""

        if output is not None:
            os.makedirs(output, exist_ok=True)

            with open(os.path.join(output, "summary.txt"), "w") as f:
                with redirect_stdout(f):
                    self.model_train.summary()
        self.model_train.summary()

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

    @staticmethod
    def ctc_complete_decoding_lambda_func(args, **arguments):
        """
        Complete CTC decoding using Tensorflow Keras (function K.ctc_decode)
        :param args:
            y_pred, input_length
        :param arguments:
            greedy, beam_width, top_paths
        :return:
            K.ctc_decode with dtype="float32"
        """

        y_pred, input_length = args
        my_params = arguments

        assert (K.backend() == "tensorflow")

        return K.cast(K.ctc_decode(y_pred, tf.squeeze(input_length), greedy=my_params["greedy"],
                      beam_width=my_params["beam_width"], top_paths=my_params["top_paths"])[0][0],
                      dtype="float32")
