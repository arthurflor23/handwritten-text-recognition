"""Build and set default callbacks to the HTR model"""

import os
import itertools
import numpy as np
# import editdistance
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint


class HTRCallback():
    """Default callbacks to HTR model"""

    def __init__(self, args, htr):
        self.output = args.output
        self.checkpoint = os.path.join(self.output, "checkpoint_weights.hdf5")
        self.logger = os.path.join(self.output, "logger.log")
        self.decoder = CTCDecoder(self.output, htr.dictionary, htr.extract_ctc_decode)

        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def get_train(self, next_val):
        """Build/Call callbacks to the model"""

        self.decoder.set_batch(next_val)
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

        return [self.decoder, logger, tensorboard, earlystopping, checkpoint]


class CTCDecoder(Callback):
    """Translate CTC output and metric calculations"""

    def __init__(self, output, dictionary, extract_ctc_decode):
        self.output = output
        self.dictionary = dictionary
        self.extract_ctc_decode = extract_ctc_decode

    def set_batch(self, next_val):
        self.text_img_gen = next_val

    def on_epoch_end(self, epoch, logs={}):
        data_batch = next(self.text_img_gen)[0]
        ctc_decode = self.extract_ctc_decode([data_batch['the_inputs'], data_batch['input_length']])[0]

        print("\n\n", ctc_decode)
        # print("\n\n", ctc_decode.shape, "\n\n")
