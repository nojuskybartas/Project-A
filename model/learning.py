# some python versions fail to load the path variables, so we're doing it manually here before importing tensorflow
import os

try:
    loaddir = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin"
    os.add_dll_directory(loaddir)
except FileNotFoundError as e:
    print(e)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Save model weights
checkpointer = ModelCheckpoint(filepath="eth_pred_model_weights.hdf5", verbose=1, save_best_only=False)
# Use early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=7)

model_metrics = [checkpointer, earlystopping]


def build_neural_model(window_len, input_columns, output_size, neurons=3000, activ_func='linear',
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
    the_model = build_neural_model(14, 1, output_size=1)
    print(the_model.summary())
