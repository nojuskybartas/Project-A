import os

try:
    # some python versions fail to load the path variables, so we're doing it manually here before importing tensorflow
    loaddir = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin"
    os.add_dll_directory(loaddir)
except FileNotFoundError as e:
    print(e)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, Activation, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model.settings import PATIENCE, VERBOSE


# Save model weights
checkpointer = ModelCheckpoint(filepath="pred_model_weights.hdf5", verbose=VERBOSE, save_best_only=True)
# Use early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='loss', mode='min', verbose=VERBOSE, patience=PATIENCE)

model_metrics = [checkpointer, earlystopping]

# This function builds a sequential model (linear - only one pathway)
def build_sequential_model(window_len, input_columns, output_size, neurons=3000, activ_func='linear',
                       dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    # model.add(Embedding(window_len, 512))
    model.add(GRU(neurons, input_shape=(window_len, input_columns)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size, activation=activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

# This function builds a nn model using the 'functional api': its more advanced and allows to have multiple pathways (non-linear)
def build_model(window_len, input_columns, output_size, neurons, activ_func='linear',
                       dropout=0.2, loss='mse', optimizer='adam'):

    inputs = keras.Input(shape=(window_len, input_columns))

    # long path
    gru = GRU(neurons)
    x = gru(inputs)
    x = Dense(neurons*2, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(neurons*2, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(neurons*2, activation="relu")(x)
    x = Dropout(dropout)(x)
    # x = Dense(neurons*2, activation="relu")(x)
    # x = Dropout(dropout)(x)
    # x = Dense(neurons, activation="relu")(x)
    # x = Dropout(dropout)(x)
    # x = Dense(neurons, activation="relu")(x)
    # x = Dropout(dropout)(x)
    # x = Dense(neurons, activation="relu")(x)
    # x = Dropout(dropout)(x)
    # x = Dense(512, activation="linear")(x)
    # x = Dropout(dropout)(x)
    # x = Dense(256, activation="linear")(x)
    # x = Dropout(dropout)(x)
    # x = Dense(128, activation="linear")(x)
    # x = Dropout(dropout)(x)
    # x = Dense(64, activation="linear")(x)
    # x = Dropout(dropout)(x)
    # x = Dense(32, activation="linear")(x)
    # x = Dropout(dropout)(x)
    # x = Dense(16, activation="linear")(x)
    # x = Dropout(dropout)(x)
    # x = Dense(output_size, activation="linear")(x)

    # # short path
    # gru = GRU(int(512))
    # x2 = gru(inputs)
    # x2 = Dense(output_size, activation="linear")(x2)

    # x = Add()([x,x2])
    # x = Activation('linear')(x)

    outputs = Dense(output_size, activation=activ_func)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Price_Prediction_Model")
    model.compile(loss=loss, optimizer=optimizer)
    return model

def load_model(model_folder, optimizer, loss, metrics=None):
    if not metrics:
        metrics = metrics
    with open(os.path.join('trained_models', model_folder, 'eth_pred_model.json'), 'r') as json_file:
        json_saved_model = json_file.read()

    # history=np.load(os.path.join('trained_models', model_folder, 'history.npy'),allow_pickle='TRUE').item()
    
    model = tf.keras.models.model_from_json(json_saved_model)
    model.load_weights(os.path.join('trained_models', model_folder, 'eth_pred_model_weights.hdf5'))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


if __name__ == '__main__':
    the_model = build_model(14, 1, output_size=1)
    print(the_model.summary())
