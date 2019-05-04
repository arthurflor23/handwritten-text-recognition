import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer, Progbar
from tensorflow.keras.layers import Lambda, TimeDistributed, Activation, Dense
from tensorflow.keras.preprocessing import sequence

"""
authors: Yann Soullard, Cyprien Ruffino (2017)
LITIS lab, university of Rouen (France)

The CTCModel class extends the Keras Model for the use of the Connectionist Temporal Classification (CTC)
One makes use of the CTC proposed in tensorflow. Thus CTCModel can only be used with the backend tensorflow.

The CTCModel structure is composed of 3 branches. Each branch is a Keras Model:
    - One for computing the CTC loss (model_train)
    - One for predicting using the ctc_decode method (model_pred)
    - One for analyzing (model_eval) that computes the Label Error Rate (LER) and Sequence Error Rate (SER).

In a Keras Model, x is the input features and y the labels.
Here, x data are of the form [input_sequences, label_sequences, inputs_lengths, labels_length]
and y are not used as in a Keras Model (this is an array which is not considered,
the labeling is given in the x data structure).
"""


class CTCModel:

    def __init__(self, inputs, outputs, greedy=True, beam_width=100, top_paths=1, charset=None):
        """
        Initialization of a CTC Model.
        :param inputs: Input layer of the neural network
            outputs: Last layer of the neural network before CTC (e.g. a TimeDistributed Dense)
            greedy, beam_width, top_paths: Parameters of the CTC decoding (see ctc decoding tensorflow for more details)
            charset: labels related to the input of the CTC approach
        """
        self.model_train = None
        self.model_pred = None
        self.model_eval = None
        if not isinstance(inputs, list):
            self.inputs = [inputs]
        else:
            self.inputs = inputs
        if not isinstance(outputs, list):
            self.outputs = [outputs]
        else:
            self.outputs = outputs

        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = top_paths
        self.charset = charset

    def compile(self, optimizer):
        """
        Configures the CTC Model for training.

        There is 3 Keras models:
            - one for training
            - one for predicting
            - one for analyzing

        Lambda layers are used to compute:
            - the CTC loss function
            - the CTC decoding
            - the CTC evaluation

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
        out_decoded_dense = Lambda(self.ctc_complete_decoding_lambda_func, output_shape=(None, None), name="CTCdecode", arguments={
                                   "greedy": self.greedy, "beam_width": self.beam_width, "top_paths": self.top_paths},
                                   dtype="float32")(self.outputs + [input_length])

        # Lambda layer to perform an analysis (CER and SER)
        out_analysis = Lambda(self.ctc_complete_analysis_lambda_func, output_shape=(None,), name="CTCanalysis", arguments={
                              "greedy": self.greedy, "beam_width": self.beam_width, "top_paths": self.top_paths},
                              dtype="float32")(self.outputs + [labels, input_length, label_length])

        # create Keras models
        self.model_init = Model(inputs=self.inputs, outputs=self.outputs)
        self.model_train = Model(inputs=self.inputs + [labels, input_length, label_length], outputs=loss_out)
        self.model_pred = Model(inputs=self.inputs + [input_length], outputs=out_decoded_dense)
        self.model_eval = Model(inputs=self.inputs + [labels, input_length, label_length], outputs=out_analysis)

        # Compile models
        self.model_train.compile(loss={"CTCloss": lambda yt, yp: yp}, optimizer=optimizer)
        self.model_pred.compile(loss={"CTCdecode": lambda yt, yp: yp}, optimizer=optimizer)
        self.model_eval.compile(loss={"CTCanalysis": lambda yt, yp: yp}, optimizer=optimizer)

    def get_model_train(self):
        """
        :return: Model used for training using the CTC approach
        """
        return self.model_train

    def get_model_pred(self):
        """
        :return: Model used for testing using the CTC approach
        """
        return self.model_pred

    def get_model_eval(self):
        """
        :return: Model used for evaluating using the CTC approach
        """
        return self.model_eval

    def get_loss_on_batch(self, inputs, verbose=False):
        """
        Computation the loss
        inputs is a list of 4 elements:
            x_features, y_label, x_len, y_len (similarly to the CTC in tensorflow)
        :return: Probabilities (output of the TimeDistributedDense layer)
        """

        x = inputs[0]
        x_len = inputs[2]
        y = inputs[1]
        y_len = inputs[3]

        no_lab = True if 0 in y_len else False

        if no_lab is False:
            loss_data = self.model_train.predict_on_batch([x, y, x_len, y_len], verbose=verbose)

        loss = np.sum(loss_data)

        return loss, loss_data

    def get_loss(self, inputs, verbose=False):
        """
        Computation the loss
        inputs is a list of 4 elements:
            x_features, y_label, x_len, y_len (similarly to the CTC in tensorflow)
        :return: Probabilities (output of the TimeDistributedDense layer)
        """

        x = inputs[0]
        x_len = inputs[2]
        y = inputs[1]
        y_len = inputs[3]
        batch_size = x.shape[0]

        no_lab = True if 0 in y_len else False

        if no_lab is False:
            loss_data = self.model_train.predict([x, y, x_len, y_len], batch_size=batch_size, verbose=verbose)

        loss = np.sum(loss_data)

        return loss, loss_data

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

        loss = np.sum(loss_per_data)
        return loss, loss_per_data

    def get_probas_generator(self, generator, nb_batchs, verbose=False):
        """
        Get the probabilities of each label at each time of an observation sequence (matrix T x D)
        This is the output of the softmax function after the recurrent layers (the input of the CTC computations)

        Computation is done in batches using a generator. This function does not exist in a Keras Model.

        :return: A set of probabilities for each sequence and each time frame, one probability per label + the blank
            (this is the output of the TimeDistributed Dense layer, the blank label is the last probability)
        """

        probs_epoch = []

        for k in range(nb_batchs):

            data = next(generator)

            x = data[0][0]
            x_len = data[0][2]
            batch_size = x.shape[0]

            # Find the output of the softmax function
            probs = self.model_init.predict(x, batch_size=batch_size, verbose=verbose)

            # Select the outputs that do not refer to padding
            probs_epoch += [np.asarray(probs[data_idx, :x_len[data_idx][0], :]) for data_idx in range(batch_size)]

        return probs_epoch

    def get_probas_on_batch(self, inputs, verbose=False):
        """
        Get the probabilities of each label at each time of an observation sequence (matrix T x D)
        This is the output of the softmax function after the recurrent layers (the input of the CTC computations)

        Computation is done for a batch. This function does not exist in a Keras Model.

        :return: A set of probabilities for each sequence and each time frame, one probability per label + the blank
            (this is the output of the TimeDistributed Dense layer, the blank label is the last probability)
        """

        x = inputs[0]
        x_len = inputs[2]
        batch_size = x.shape[0]

        #  Find the output of the softmax function
        probs = self.model_init.predict(x, batch_size=batch_size, verbose=verbose)

        # Select the outputs that do not refer to padding
        probs_epoch = [np.asarray(probs[data_idx, :x_len[data_idx][0], :]) for data_idx in range(batch_size)]

        return probs_epoch

    def get_probas(self, inputs, batch_size, verbose=False):
        """
        Get the probabilities of each label at each time of an observation sequence (matrix T x D)
        This is the output of the softmax function after the recurrent layers (the input of the CTC computations)

        Computation is done for a batch. This function does not exist in a Keras Model.

        :return: A set of probabilities for each sequence and each time frame, one probability per label + the blank
            (this is the output of the TimeDistributed Dense layer, the blank label is the last probability)
        """

        x = inputs[0]
        x_len = inputs[2]

        #  Find the output of the softmax function
        probs = self.model_init.predict(x, batch_size=batch_size, verbose=verbose)

        # Select the outputs that do not refer to padding
        probs_epoch = [np.asarray(probs[data_idx, :x_len[data_idx][0], :]) for data_idx in range(batch_size)]

        return probs_epoch

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
                      initial_epoch=0):
        """
        Model training on data yielded batch-by-batch by a Python generator.

        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.

        A major modification concerns the generator that must provide x data of the form:
          [input_sequences, label_sequences, inputs_lengths, labels_length]
        (in a similar way than for using CTC in tensorflow)

        :param: See keras.engine.Model.fit_generator()
        :return: A History object
        """
        out = self.model_train.fit_generator(generator, steps_per_epoch, epochs=epochs, verbose=verbose,
                                             callbacks=callbacks, validation_data=validation_data,
                                             validation_steps=validation_steps, class_weight=class_weight,
                                             max_queue_size=max_queue_size, workers=workers,
                                             initial_epoch=initial_epoch)

        self.model_pred.set_weights(self.model_train.get_weights())  # required??
        self.model_eval.set_weights(self.model_train.get_weights())
        return out

    def fit(self, x=None,
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
            validation_steps=None):
        """
        Model training on data.

        A major modification concerns the x input of the form:
          [input_sequences, label_sequences, inputs_lengths, labels_length]
        (in a similar way than for using CTC in tensorflow)

        :param: See keras.engine.Model.fit()
        :return: A History object
        """

        out = self.model_train.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                                   callbacks=callbacks, validation_split=validation_split, validation_data=validation_data,
                                   shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight,
                                   initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

        self.model_pred.set_weights(self.model_train.get_weights())
        self.model_eval.set_weights(self.model_train.get_weights())

        return out

    def train_on_batch(self, x, y, sample_weight=None, class_weight=None):
        """ Runs a single gradient update on a single batch of data.
        See Keras.Model for more details.
        """

        out = self.model_train.train_on_batch(x, y, sample_weight=sample_weight, class_weight=class_weight)

        self.model_pred.set_weights(self.model_train.get_weights())
        self.model_eval.set_weights(self.model_train.get_weights())

        return out

    def evaluate(self, x=None, batch_size=None, verbose=1, steps=None, metrics=["loss", "ler", "ser"]):
        """ Evaluates the model on a dataset_manager.

                :param: See keras.engine.Model.predict()
                :return: A History object

                CTC evaluation on data yielded batch-by-batch by a Python generator.

                Inputs x:
                        x_input = Input data as a 3D Tensor (batch_size, max_input_len, dim_features)
                        y = Input data as a 2D Tensor (batch_size, max_label_len)
                        x_len = 1D array with the length of each data in batch_size
                        y_len = 1D array with the length of each labeling

                metrics = list of metrics that are computed. This is elements among the 3 following metrics:
                    "loss" : compute the loss function on x
                    "ler" : compute the label error rate
                    "ser" : compute the sequence error rate

                Outputs: a list containing:
                    ler_dataset = label error rate for each data (a list)
                    seq_error = sequence error rate on the dataset_manager
        """
        seq_error = 0

        x_input = x[0]
        x_len = x[2]
        y = x[1]
        y_len = x[3]
        nb_data = x_input.shape[0]

        if "ler" in metrics or "ser" in metrics:
            eval_batch = self.model_eval.predict([x_input, y, x_len, y_len], batch_size=batch_size, verbose=verbose, steps=steps)

        if "ser" in metrics:
            seq_error += np.sum([1 for ler_data in eval_batch if ler_data != 0])
            seq_error = seq_error / nb_data if nb_data > 0 else -1.

        outmetrics = []
        if "loss" in metrics:
            outmetrics.append(self.get_loss(x))
        if "ler" in metrics:
            outmetrics.append(eval_batch)
        if "ser" in metrics:
            outmetrics.append(seq_error)

        return outmetrics

    def test_on_batch(self, x=None, metrics=["loss", "ler", "ser"]):
        """ Name of a Keras Model function: this relates to evaluate on batch """
        return self.evaluate_on_batch(x)

    def evaluate_on_batch(self, x=None, metrics=["loss", "ler", "ser"]):
        """ Evaluates the model on a dataset_manager.

                :param: See keras.engine.Model.predict_on_batch()
                :return: A History object

                CTC evaluation on data yielded batch-by-batch by a Python generator.

                Inputs x:
                        x_input = Input data as a 3D Tensor (batch_size, max_input_len, dim_features)
                        y = Input data as a 2D Tensor (batch_size, max_label_len)
                        x_len = 1D array with the length of each data in batch_size
                        y_len = 1D array with the length of each labeling

                metrics = list of metrics that are computed. This is elements among the 3 following metrics:
                    "loss" : compute the loss function on x
                    "ler" : compute the label error rate
                    "ser" : compute the sequence error rate

                Outputs: a list containing:
                    ler_dataset = label error rate for each data (a list)
                    seq_error = sequence error rate on the dataset_manager
        """
        seq_error = 0

        x_input = x[0]
        x_len = x[2]
        y = x[1]
        y_len = x[3]
        nb_data = x_input.shape[0]

        if "ler" in metrics or "ser" in metrics:
            eval_batch = self.model_eval.predict_on_batch([x_input, y, x_len, y_len])

        if "ser" in metrics:
            seq_error += np.sum([1 for ler_data in eval_batch if ler_data != 0])
            seq_error = seq_error / nb_data if nb_data > 0 else -1.

        outmetrics = []
        if "loss" in metrics:
            outmetrics.append(self.get_loss(x))
        if "ler" in metrics:
            outmetrics.append(eval_batch)
        if "ser" in metrics:
            outmetrics.append(seq_error)

        return outmetrics

    def evaluate_generator(self, generator, steps=None, max_queue_size=10, workers=1,
                           verbose=False, metrics=["loss", "ler", "ser"]):
        """ Evaluates the model on a data generator.

        :param: See keras.engine.Model.fit()
        :return: A History object

        CTC evaluation on data yielded batch-by-batch by a Python generator.

        Inputs:
            generator = DataGenerator class that returns:
                    x = Input data as a 3D Tensor (batch_size, max_input_len, dim_features)
                    y = Input data as a 2D Tensor (batch_size, max_label_len)
                    x_len = 1D array with the length of each data in batch_size
                    y_len = 1D array with the length of each labeling
            nb_batchs = number of batchs that are evaluated

            metrics = list of metrics that are computed. This is elements among the 3 following metrics:
                    "loss" : compute the loss function on x
                    "ler" : compute the label error rate
                    "ser" : compute the sequence error rate
            Warning: if the "loss" and another metric are requested, make sure that the number of
                     steps allows to evaluate the entire dataset, even if the data given by the generator
                     will be not the same for all metrics. To make sure, you can only compute
                     "ler" and "ser" here then initialize again the generator and call get_loss_generator.

        Outputs: a list containing the metrics given in argument:
            loss : the loss on the set
            ler : the label error rate for each data (a list)
            seq_error : the sequence error rate on the dataset
        """

        if "ler" in metrics or "ser" in metrics:
            ler_dataset = self.model_eval.predict_generator(generator, steps,
                                                            max_queue_size=max_queue_size, workers=workers,
                                                            use_multiprocessing=False, verbose=verbose)
        if "ser" in metrics:
            seq_error = float(np.sum([1 for ler_data in ler_dataset if ler_data != 0])) \
                / len(ler_dataset) if len(ler_dataset) > 0 else 1.

        outmetrics = []
        if "loss" in metrics:
            outmetrics.append(self.get_loss_generator(generator, steps))
        if "ler" in metrics:
            outmetrics.append(ler_dataset)
        if "ser" in metrics:
            outmetrics.append(seq_error)

        return outmetrics

    def predict_on_batch(self, x):
        """Returns predictions for a single batch of samples.

                # Arguments
                    x: [Input samples as a Numpy array, Input length as a numpy array]

                # Returns
                    Numpy array(s) of predictions.
        """

        # batch_size = x[0].shape[0]
        # return self.predict(x, batch_size=batch_size)

        out = self.model_pred.predict_on_batch(x)
        output = [[pr for pr in pred if pr != -1] for pred in out]

        return output

    def predict_generator(self, generator, steps,
                          max_queue_size=10,
                          workers=1,
                          verbose=0,
                          decode_func=None):
        """Generates predictions for the input samples from a data generator.

        The generator should return the same kind of data as accepted by
        `predict_on_batch`.

        generator = DataGenerator class that returns:
                        x = Input data as a 3D Tensor (batch_size, max_input_len, dim_features)
                        y = Input data as a 2D Tensor (batch_size, max_label_len)
                        x_len = 1D array with the length of each data in batch_size
                        y_len = 1D array with the length of each labeling

        # Arguments
            generator: Generator yielding batches of input samples
                    or an instance of Sequence (keras.utils.Sequence)
                    object in order to avoid duplicate data
                    when using multiprocessing.
            steps: Total number of steps (batches of samples)
                to yield from `generator` before stopping.
            max_queue_size: Maximum size for the generator queue.
            workers: Maximum number of processes to spin up
                when using process based threading
            verbose: verbosity mode, 0 or 1.
            decode_func: a function for decoding a list of predicted sequences (using self.charset)

        # Returns
            A tuple containing:
                A numpy array(s) of ground truth.
                A numpy array(s) of predictions.

        # Raises
            ValueError: In case the generator yields
                data in an invalid format.
        """
        self.model_pred._make_predict_function()

        steps_done = 0
        all_outs = []
        all_lab = []

        is_sequence = isinstance(generator, Sequence)
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
                    # Compatibility with the generators
                    # used for training.
                    if len(generator_output) == 2:
                        x, _ = generator_output
                    elif len(generator_output) == 3:
                        x, _, _ = generator_output
                    else:
                        raise ValueError("Output of generator should be "
                                         "a tuple `(x, y, sample_weight)` "
                                         "or `(x, y)`. Found: " + str(generator_output))
                else:
                    # Assumes a generator that only
                    # yields inputs (not targets and sample weights).
                    x = generator_output

                [x_input, y, x_length, y_length] = x
                outs = self.predict_on_batch([x_input, x_length])

                if not isinstance(outs, list):
                    outs = [outs]

                if not all_outs:
                    for out in outs:
                        all_outs.append([])
                        all_lab.append([])

                for i, out in enumerate(outs):
                    all_outs[i].append([val_out for val_out in out if val_out != -1])
                    if isinstance(y_length[i], list):
                        all_lab[i].append(y[i][:y_length[i][0]])
                    elif isinstance(y_length[i], int):
                        all_lab[i].append(y[i][:y_length[i]])
                    elif isinstance(y_length[i], float):
                        all_lab[i].append(y[i][:int(y_length[i])])
                    else:
                        all_lab[i].append(y[i])

                steps_done += 1
                if verbose == 1:
                    progbar.update(steps_done)

        finally:
            if enqueuer is not None:
                enqueuer.stop()

        batch_size = len(all_outs)
        nb_data = len(all_outs[0])
        pred_out = []
        lab_out = []
        for i in range(nb_data):
            pred_out += [all_outs[b][i] for b in range(batch_size)]
            lab_out += [all_lab[b][i] for b in range(batch_size)]

        if decode_func is not None:  # convert model prediction (a label between 0 to nb_labels to an original label sequence)
            lab_out = decode_func(lab_out, self.charset)
            pred_out = decode_func(pred_out, self.charset)

        return lab_out, pred_out

    def summary(self):
        return self.model_train.summary()

    def predict(self, x, batch_size=None, verbose=0, steps=None, max_len=None, max_value=999):
        """
        The same function as in the Keras Model but with a different function predict_loop for dealing with variable length predictions
        Except that x = [x_features, x_len]

        Generates output predictions for the input samples.

                Computation is done in batches.

                # Arguments
                    x: The input data, as a Numpy array
                        (or list of Numpy arrays if the model has multiple outputs).
                    batch_size: Integer. If unspecified, it will default to 32.
                    verbose: Verbosity mode, 0 or 1.
                    steps: Total number of steps (batches of samples)
                        before declaring the prediction round finished.
                        Ignored with the default value of `None`.

                # Returns
                    Numpy array(s) of predictions.

                # Raises
                    ValueError: In case of mismatch between the provided
                        input data and the model"s expectations,
                        or in case a stateful model receives a number of samples
                        that is not a multiple of the batch size.
                """
        [x_inputs, x_len] = x

        if max_len is None:
            max_len = np.max(x_len)

        # Backwards compatibility.
        if batch_size is None and steps is None:
            batch_size = 32
        if x is None and steps is None:
            raise ValueError("If predicting from data tensors, "
                             "you should specify the `steps` "
                             "argument.")
        # Validate user data.
        x = _standardize_input_data(x, self.model_pred._feed_input_names,
                                    self.model_pred._feed_input_shapes,
                                    check_batch_axis=False)
        if self.model_pred.stateful:
            if x[0].shape[0] > batch_size and x[0].shape[0] % batch_size != 0:
                raise ValueError("In a stateful network, "
                                 "you should only pass inputs with "
                                 "a number of samples that can be "
                                 "divided by the batch size. Found: " + str(x[0].shape[0]) + " samples. "
                                 "Batch size: " + str(batch_size) + ".")

        # Prepare inputs, delegate logic to `_predict_loop`.
        # if self.model_pred.uses_learning_phase and not isinstance(K.learning_phase(), int):
        if not isinstance(K.learning_phase(), int):
            ins = x + [0.]
        else:
            ins = x
        self.model_pred._make_predict_function()
        f = self.model_pred.predict_function
        out = self._predict_loop(f, ins, batch_size=batch_size, max_value=max_value,
                                 verbose=verbose, steps=steps, max_len=max_len)

        out_decode = [dec_data[:list(dec_data).index(max_value)] if max_value in dec_data else dec_data for i, dec_data
                      in enumerate(out)]
        return out_decode

    def _predict_loop(self, f, ins, max_len=100, max_value=999, batch_size=32, verbose=0, steps=None):
        """Abstract method to loop over some data in batches.

        Keras function that has been modified.

        # Arguments
            f: Keras function returning a list of tensors.
            ins: list of tensors to be fed to `f`.
            batch_size: integer batch size.
            verbose: verbosity mode.
            steps: Total number of steps (batches of samples)
                before declaring `_predict_loop` finished.
                Ignored with the default value of `None`.

        # Returns
            Array of predictions (if the model has a single output)
            or list of arrays of predictions
            (if the model has multiple outputs).
        """
        # num_samples = self.model_pred._check_num_samples(ins, batch_size,
        #                                       steps,
        #                                       "steps")
        num_samples = check_num_samples(ins,
                                        batch_size=batch_size,
                                        steps=steps,
                                        steps_name="steps")

        if steps is not None:
            # Step-based predictions.
            # Since we do not know how many samples
            # we will see, we cannot pre-allocate
            # the returned Numpy arrays.
            # Instead, we store one array per batch seen
            # and concatenate them upon returning.
            unconcatenated_outs = []
            for step in range(steps):
                batch_outs = f(ins)
                if not isinstance(batch_outs, list):
                    batch_outs = [batch_outs]
                if step == 0:
                    for batch_out in batch_outs:
                        unconcatenated_outs.append([])
                for i, batch_out in enumerate(batch_outs):
                    unconcatenated_outs[i].append(batch_out)

            if len(unconcatenated_outs) == 1:
                return np.concatenate(unconcatenated_outs[0], axis=0)
            return [np.concatenate(unconcatenated_outs[i], axis=0)
                    for i in range(len(unconcatenated_outs))]
        else:
            # Sample-based predictions.
            outs = []
            batches = _make_batches(num_samples, batch_size)
            index_array = np.arange(num_samples)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                if ins and isinstance(ins[-1], float):
                    # Do not slice the training phase flag.
                    ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
                else:
                    ins_batch = _slice_arrays(ins, batch_ids)
                batch_outs = f(ins_batch)
                if not isinstance(batch_outs, list):
                    batch_outs = [batch_outs]
                if batch_index == 0:
                    # Pre-allocate the results arrays.
                    for batch_out in batch_outs:
                        shape = (num_samples, max_len)
                        outs.append(np.zeros(shape, dtype=batch_out.dtype))
                for i, batch_out in enumerate(batch_outs):
                    outs[i][batch_start:batch_end] = sequence.pad_sequences(batch_out, value=float(max_value),
                                                                            maxlen=max_len,
                                                                            dtype=batch_out.dtype, padding="post")

            if len(outs) == 1:
                return outs[0]
            return outs

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
        # , ignore_longer_outputs_than_inputs=True)

    @staticmethod
    def ctc_complete_decoding_lambda_func(args, **arguments):
        """
        Complete CTC decoding using Keras (function K.ctc_decode)
        :param args:
            y_pred, input_length
        :param arguments:
            greedy, beam_width, top_paths
        :return:
            K.ctc_decode with dtype="float32"
        """

        # import tensorflow as tf # Require for loading a model saved

        y_pred, input_length = args
        my_params = arguments

        assert (K.backend() == "tensorflow")

        return K.cast(K.ctc_decode(y_pred, tf.squeeze(input_length), greedy=my_params["greedy"],
                      beam_width=my_params["beam_width"], top_paths=my_params["top_paths"])[0][0],
                      dtype="float32")

    @staticmethod
    def ctc_complete_analysis_lambda_func(args, **arguments):
        """
        Complete CTC analysis using Keras and tensorflow
        WARNING : tf is required
        :param args:
            y_pred, labels, input_length, label_len
        :param arguments:
            greedy, beam_width, top_paths
        :return:
            ler = label error rate
        """

        # import tensorflow as tf # Require for loading a model saved

        y_pred, labels, input_length, label_len = args
        my_params = arguments

        assert (K.backend() == "tensorflow")

        batch = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)
        input_length = tf.cast(tf.squeeze(input_length), tf.int32)

        greedy = my_params["greedy"]
        beam_width = my_params["beam_width"]
        top_paths = my_params["top_paths"]

        if greedy:
            (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
                inputs=batch,
                sequence_length=input_length)
        else:
            (decoded, log_prob) = tf.nn.ctc_beam_search_decoder(
                inputs=batch, sequence_length=input_length,
                beam_width=beam_width, top_paths=top_paths)

        cast_decoded = tf.cast(decoded[0], tf.float32)

        sparse_y = K.ctc_label_dense_to_sparse(labels, tf.cast(tf.squeeze(label_len), tf.int32))
        ed_tensor = tf_edit_distance(cast_decoded, sparse_y, norm=True)
        ler_per_seq = Kreshape_To1D(ed_tensor)

        return K.cast(ler_per_seq, dtype="float32")

    def save_model(self, path_dir, charset=None):
        """ Save a model in path_dir
        save model_train, model_pred and model_eval in json
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

        model_json = self.model_eval.to_json()
        with open(path_dir + "/model_eval.json", "w") as json_file:
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
        load model_train, model_pred and model_eval from hdf5
        """

        self.model_train.load_weights(checkpoint)
        self.model_pred.set_weights(self.model_train.get_weights())
        self.model_eval.set_weights(self.model_train.get_weights())

    def load_model(self, path_dir, optimizer, init_archi=True, file_weights=None, change_parameters=False,
                   init_last_layer=False, add_layers=None, trainable=False, removed_layers=2):
        """ Load a model in path_dir
        load model_train, model_pred and model_eval from json
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

            json_file = open(path_dir + "/model_eval.json", "r")
            loaded_model_json = json_file.read()
            json_file.close()
            self.model_eval = model_from_json(loaded_model_json, custom_objects={"tf": tf, "ctc": tf.nn,
                                                                                 "tf_edit_distance": tf_edit_distance,
                                                                                 "Kreshape_To1D": Kreshape_To1D})

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
                self.model_eval.set_weights(self.model_train.get_weights())
            elif os.path.exists(path_dir + file_weights):
                self.model_train.load_weights(path_dir + file_weights)
                self.model_pred.set_weights(self.model_train.get_weights())
                self.model_eval.set_weights(self.model_train.get_weights())

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

            # Lambda layer to perform an analysis (CER and SER)
            out_analysis = Lambda(self.ctc_complete_analysis_lambda_func, output_shape=(None,), name="CTCanalysis",
                                  arguments={"greedy": self.greedy,
                                             "beam_width": self.beam_width, "top_paths": self.top_paths},
                                  dtype="float32")(
                self.outputs + [labels, input_length, label_length])

            # create Keras models
            self.model_init = Model(inputs=self.inputs, outputs=self.outputs)
            self.model_train = Model(inputs=self.inputs + [labels, input_length, label_length], outputs=loss_out)
            self.model_pred = Model(inputs=self.inputs + [input_length], outputs=out_decoded_dense)
            self.model_eval = Model(inputs=self.inputs + [labels, input_length, label_length], outputs=out_analysis)

            # Compile models
            self.model_train.compile(loss={"CTCloss": lambda yt, yp: yp}, optimizer=optimizer)
            self.model_pred.compile(loss={"CTCdecode": lambda yt, yp: yp}, optimizer=optimizer)
            self.model_eval.compile(loss={"CTCanalysis": lambda yt, yp: yp}, optimizer=optimizer)

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

            # Lambda layer to perform an analysis (CER and SER)
            out_analysis = Lambda(self.ctc_complete_analysis_lambda_func, output_shape=(None,), name="CTCanalysis",
                                  arguments={"greedy": self.greedy,
                                             "beam_width": self.beam_width, "top_paths": self.top_paths},
                                  dtype="float32")(
                self.outputs + [labels, input_length, label_length])

            # create Keras models
            self.model_init = Model(inputs=self.inputs, outputs=self.outputs)
            self.model_train = Model(inputs=self.inputs + [labels, input_length, label_length], outputs=loss_out)
            self.model_pred = Model(inputs=self.inputs + [input_length], outputs=out_decoded_dense)
            self.model_eval = Model(inputs=self.inputs + [labels, input_length, label_length], outputs=out_analysis)

            # Compile models
            self.model_train.compile(loss={"CTCloss": lambda yt, yp: yp}, optimizer=optimizer)
            self.model_pred.compile(loss={"CTCdecode": lambda yt, yp: yp}, optimizer=optimizer)
            self.model_eval.compile(loss={"CTCanalysis": lambda yt, yp: yp}, optimizer=optimizer)


def _standardize_input_data(data, names, shapes=None,
                            check_batch_axis=True,
                            exception_prefix=""):
    """Normalizes inputs and targets provided by users.

    Users may pass data as a list of arrays, dictionary of arrays,
    or as a single array. We normalize this to an ordered list of
    arrays (same order as `names`), while checking that the provided
    arrays have shapes that match the network"s expectations.

    # Arguments
        data: User-provided input data (polymorphic).
        names: List of expected array names.
        shapes: Optional list of expected array shapes.
        check_batch_axis: Boolean; whether to check that
            the batch axis of the arrays matches the expected
            value found in `shapes`.
        exception_prefix: String prefix used for exception formatting.

    Keras function that has been modified.

    # Returns
        List of standardized input arrays (one array per model input).

    # Raises
        ValueError: in case of improperly formatted user-provided data.
    """
    if not names:
        if data is not None and hasattr(data, "__len__") and len(data):
            raise ValueError("Error when checking model " + exception_prefix + ": expected no data, but got:", data)
        return []
    if data is None:
        return [None for _ in range(len(names))]
    if isinstance(data, dict):
        arrays = []
        for name in names:
            if name not in data:
                raise ValueError("No data provided for '" + name + "'. "
                                 "Need data for each key in: " + str(names))
            arrays.append(data[name])
    elif isinstance(data, list):
        if len(data) != len(names):
            if data and hasattr(data[0], "shape"):
                raise ValueError("Error when checking model " + exception_prefix + ": the list of Numpy arrays "
                                 "that you are passing to your model is not the size the model expected. "
                                 "Expected to see " + str(len(names)) + " array(s), but instead got "
                                 "the following list of " + str(len(data)) + " arrays: " + str(data)[:200] + "...")
            else:
                if len(names) == 1:
                    data = [np.asarray(data)]
                else:
                    raise ValueError(
                        "Error when checking model " + exception_prefix + ": you are passing a list as "
                        "input to your model, but the model expects "
                        "a list of " + str(len(names)) + " Numpy arrays instead. "
                        "The list you passed was: " + str(data)[:200])
        arrays = data
    else:
        if not hasattr(data, "shape"):
            raise TypeError("Error when checking model " + exception_prefix + ": data should be a Numpy array, "
                            "or list/dict of Numpy arrays. Found: " + str(data)[:200] + "...")
        if len(names) > 1:
            # Case: model expects multiple inputs but only received
            # a single Numpy array.
            raise ValueError("The model expects " + str(len(names)) + " " + exception_prefix + " arrays, but only "
                             "received one array. Found: array with shape " + str(data.shape))
        arrays = [data]

    # Make arrays at least 2D.
    for i in range(len(names)):
        array = arrays[i]
        if len(array.shape) == 1:
            array = np.expand_dims(array, 1)
            arrays[i] = array

    # Check shapes compatibility.
    if shapes:
        for i in range(len(names)):
            if shapes[i] is None:
                continue
            array = arrays[i]
            if len(array.shape) != len(shapes[i]):
                raise ValueError("Error when checking " + exception_prefix + ": "
                                 "expected " + names[i] + " to have " + str(len(shapes[i])) + " "
                                 "dimensions, but got array with shape " + str(array.shape))
            for j, (dim, ref_dim) in enumerate(zip(array.shape, shapes[i])):
                if not j and not check_batch_axis:
                    # skip the first axis
                    continue
                if ref_dim:
                    if ref_dim != dim:
                        raise ValueError(
                            "Error when checking " + exception_prefix + ": expected " + names[i] + " to have "
                            "shape " + str(shapes[i]) + " but got array with shape " + str(array.shape))
    return arrays


def _slice_arrays(arrays, start=None, stop=None):
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `_slice_arrays(x, indices)`

    Keras function that has been modified.

    # Arguments
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    # Returns
        A slice of the array(s).
    """
    if arrays is None:
        return [None]
    elif isinstance(arrays, list):
        if hasattr(start, "__len__"):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, "shape"):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, "__len__"):
            if hasattr(start, "shape"):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, "__getitem__"):
            return arrays[start:stop]
        else:
            return [None]


def _make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).

    Keras function that has been modified.

    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.

    # Returns
        A list of tuples of array indices.
    """
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]


def Kreshape_To1D(my_tensor):
    """ Reshape to a 1D Tensor using K.reshape"""

    sum_shape = K.sum(K.shape(my_tensor))
    return K.reshape(my_tensor, (sum_shape,))


def tf_edit_distance(hypothesis, truth, norm=False):
    """ Edit distance using tensorflow

    inputs are tf.Sparse_tensors """

    return tf.edit_distance(hypothesis, truth, normalize=norm, name="edit_distance")


def check_num_samples(ins,
                      batch_size=None,
                      steps=None,
                      steps_name="steps"):
    """Checks the number of samples provided for training and evaluation.
    The number of samples is not defined when running with `steps`,
    in which case the number of samples is set to `None`.
    # Arguments
        ins: List of tensors to be fed to the Keras function.
        batch_size: Integer batch size or `None` if not defined.
        steps: Total number of steps (batches of samples)
            before declaring `predict_loop` finished.
            Ignored with the default value of `None`.
        steps_name: The public API"s parameter name for `steps`.
    # Raises
        ValueError: when `steps` is `None` and the attribute `ins.shape`
        does not exist. Also raises ValueError when `steps` is not `None`
        and `batch_size` is not `None` because they are mutually
        exclusive.
    # Returns
        When `steps` is `None`, returns the number of samples to be
        processed based on the size of the first dimension of the
        first input Numpy array. When `steps` is not `None` and
        `batch_size` is `None`, returns `None`.
    # Raises
        ValueError: In case of invalid arguments.
    """
    if steps is not None and batch_size is not None:
        raise ValueError(
            "If " + steps_name + " is set, the `batch_size` must be None.")

    if not ins or any(tf.is_tensor(x) for x in ins):
        if steps is None:
            raise ValueError(
                "If your data is in the form of symbolic tensors, "
                "you should specify the `" + steps_name + "` argument "
                                                          "(instead of the `batch_size` argument, "
                                                          "because symbolic tensors are expected to produce "
                                                          "batches of input data).")
        return None

    if hasattr(ins[0], "shape"):
        return int(ins[0].shape[0])
    return None  # Edge case where ins == [static_learning_phase]
