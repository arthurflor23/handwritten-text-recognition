from tensorflow.keras.layers import TimeDistributed, Activation, Dense, Input, Bidirectional, LSTM, Masking, GaussianNoise
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from network.ctc_model import CTCModel
import os


class HTRNetwork:

    def __init__(self, dtgen):
        self.create_network(dtgen.nb_features, dtgen.dictionary, dtgen.padding_value)
        # self.callbacks()

    def create_network(self, nb_features, nb_labels, padding_value):

        # Define the network architecture
        input_data = Input(name="input", shape=(None, nb_features))

        masking = Masking(mask_value=padding_value)(input_data)
        noise = GaussianNoise(0.01)(masking)
        blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(noise)
        blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
        blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)

        dense = TimeDistributed(Dense(len(nb_labels) + 1, name="dense"))(blstm)
        outrnn = Activation("softmax", name="softmax")(dense)

        self.model = CTCModel(inputs=[input_data], outputs=[outrnn], greedy=False, beam_width=100, top_paths=1, charset=nb_labels)
        self.model.compile(Adamax(lr=0.0001))

    # def callbacks(self):
    #     output = "../output/temp/"

    #     # if os.path.exists(output + "models"):
    #     #     self.model.load_model(path_dir=output + "models", optimizer=Adamax(lr=0.0001))
    #     # else:
    #     #     os.makedirs(output + "models")

    #     tensorboard = TensorBoard(
    #         log_dir=output,
    #         histogram_freq=1,
    #         profile_batch=0,
    #         write_graph=True,
    #         write_images=True,
    #         update_freq="epoch")

    #     earlystopping = EarlyStopping(
    #         monitor="val_loss",
    #         min_delta=1e-5,
    #         patience=5,
    #         restore_best_weights=True,
    #         verbose=1)

    #     self.callbacks = [tensorboard, earlystopping]
