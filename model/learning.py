# some python versions fail to load the path variables, so we're doing it manually here before importing tensorflow
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, Embedding, Bidirectional, GRU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.eager.context import check_alive
import numpy as np


# Save model weights
checkpointer = ModelCheckpoint(filepath="eth_pred_model_weights.hdf5", verbose=1, save_best_only=False)
# Use early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=7)

metrics = [checkpointer, earlystopping]


def build_lstm_model(window_len, input_columns, output_size, neurons=3000, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    # model.add(Embedding(window_len, 512))
    model.add(GRU(neurons, input_shape =(window_len, input_columns)))
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

def load_model(model_folder, optimizer, loss, metrics=metrics):
    with open(os.path.join('trained_models', model_folder, 'eth_pred_model.json'), 'r') as json_file:
        json_savedModel= json_file.read()

    # history=np.load(os.path.join('trained_models', model_folder, 'history.npy'),allow_pickle='TRUE').item()
    
    model = tf.keras.models.model_from_json(json_savedModel)
    model.load_weights(os.path.join('trained_models', model_folder,'eth_pred_model_weights.hdf5'))
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    return model

if __name__=='__main__':
    model = build_lstm_model(14, 1)
    print(model.summary())