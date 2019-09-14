"""Define callbacks for reporting and managing the model training"""

import os
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def setup(logdir, hdf5_target):
    """Setup the list of callbacks for the model"""

    os.makedirs(os.path.join(logdir), exist_ok=True)

    callbacks = [
        CSVLogger(
            filename=os.path.join(logdir, "epochs.log"),
            separator=';',
            append=True),
        TensorBoard(
            log_dir=logdir,
            histogram_freq=10,
            profile_batch=0,
            write_graph=True,
            write_images=False,
            update_freq="epoch"),
        ModelCheckpoint(
            filepath=os.path.join(logdir, hdf5_target),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1),
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=20,
            restore_best_weights=True,
            verbose=1),
        ReduceLROnPlateau(
            monitor="val_loss",
            min_delta=0.0001,
            factor=0.2,
            patience=10,
            verbose=1)
    ]

    return callbacks
