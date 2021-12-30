import os
import logging

# (windows only for now)
if os.name == 'nt':
    try:
        logging.info('Looking for CUDA and adding it to path...')
        # some python versions fail to load the path variables, so we're doing it manually here before importing tf
    
        loaddir = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin"
        os.add_dll_directory(loaddir)
        logging.info('Found!')

    except Exception as e:
        logging.info(f'CUDA not found, this gon be slow af \n{e}')

import shutil
import json
import numpy as np
logging.info('Loading TensorFlow Libs...')  # noqa: E402
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logging.info('Done!')

# Save model weights
VERBOSE: int = 1  # 0: no visual feedback, 1: animated progress bar, 2: show number of epoch
checkpointer = ModelCheckpoint(
    filepath="pred_model_weights.hdf5", verbose=VERBOSE, save_best_only=True)
# Use early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
PATIENCE: int = 20  # For early-stopping
earlystopping = EarlyStopping(
    monitor='loss', mode='min', verbose=VERBOSE, patience=PATIENCE)

model_metrics = [checkpointer, earlystopping]

# This function builds a sequential model (linear - only one pathway)


def build_sequential(window_len, input_columns, output_size, neurons=3000, activ_func='linear',
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

# This function builds a nn model using the 'functional api':
# it's more advanced and allows to have multiple pathways (non-linear)


def build(window_len, input_columns, output_size, neurons, activ_func='linear',
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

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name="Price_Prediction_Model")
    model.compile(loss=loss, optimizer=optimizer)
    return model


def save(model, history, config):
    model_json = model.to_json()
    with open(f"{config.CURRENCY}_pred_model.json", "w") as json_file:
        json_file.write(model_json)

    if history is not None:
        np.save('history.npy', history.history)

    path = os.path.join('trained_models', config.MODEL_FOLDER)
    os.mkdir(path) if not os.path.isdir(path) else [logging.info(
        f'This model version already exists! OVERWRITING!'), shutil.rmtree(path), os.mkdir(path)]
    params = {'optimizer': config.OPTIMIZER, 'loss': config.LOSS}
    with open(os.path.join(path, 'params.json'), 'w') as json_file:
        json.dump(params, json_file)
    os.rename("history.npy", os.path.join(path, "history.npy"))
    os.rename(f"{config.CURRENCY}_pred_model.json", os.path.join(
        path, f"{config.CURRENCY}_pred_model.json"))
    os.rename("pred_model_weights.hdf5", os.path.join(
        path, f"{config.CURRENCY}_pred_model_weights.hdf5"))
    logging.info(f'Trained model successfully saved to {path}')


def load(model_folder, optimizer=None, loss=None, metrics=None):
    logging.info(f'Loading model <{model_folder}>...')
    try:
        with open(os.path.join('trained_models', model_folder, 'eth_pred_model.json'), 'r') as json_file:
            json_saved_model = json_file.read()
    except Exception as e:
        logging.info(f'Failed to load the model - model not found!\n{e}')
        return

    model = tf.keras.models.model_from_json(json_saved_model)

    # history=np.load(os.path.join('trained_models', model_folder, 'history.npy'),allow_pickle='TRUE').item()
    if optimizer is None or loss is None:
        logging.info(
            f'No model parameters given - reading from file')
        try:
            with open(os.path.join('trained_models', model_folder, 'params.json')) as json_file:
                params = json.load(json_file)
        except Exception as e:
            logging.info(f'Failed to load the params - file not found! Please provide the optimizer and loss '
                         f'in the function call or in a params.json file\n{e}')
            return
        optimizer = params['optimizer']
        loss = params['loss']
    
    try:
        logging.info('Loading model weights...')
        model.load_weights(os.path.join(
            'trained_models', model_folder, 'eth_pred_model_weights.hdf5'))
    except Exception as e:
        logging.info(f'Failed to load the model - weight file not found!\n{e}')
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logging.info(f'Successfully loaded the model')
    return model
