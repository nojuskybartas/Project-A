import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.eager.context import check_alive

checkpointer = ModelCheckpoint(filepath="eth_pred_model_weights.hdf5", verbose=1, save_best_only=False)

def build_lstm_model(window_len, output_size, neurons=100, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam', metrics=[""]):
    model = Sequential()
    model.add(LSTM(neurons, input_shape =(window_len, 1)))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

def load_model(optimizer, loss, metrics=[checkpointer]):
    with open('eth_pred_model.json', 'r') as json_file:
        json_savedModel= json_file.read()
    
    model = tf.keras.models.model_from_json(json_savedModel)
    model.load_weights('eth_pred_model_weights.hdf5')
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    return model