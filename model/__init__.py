import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from os.path import isfile, isdir
import os
import shutil
from termcolor import colored
from dataframe import DataFrame

DATA_FILENAME = 'data/ETH-USD_2016-01-01_UTC2021-11-07_UTC.data'
DATA_FILENAME_TEST = 'data/ETH-USD_2021-11-07_UTC2021-11-24_UTC.data'
assert(isfile(DATA_FILENAME))
np.random.seed(42)
WINDOW_LEN = 7
INPUT_COLUMNS = 3 # columns in the dataframe used for training
TEST_SIZE = 0.15
ZERO_BASE = False
GRU_NEURONS = 3200
EPOCHS = 250 # big number, because we have earlystopping
BATCH_SIZE = 32
LOSS = 'mse'
DROPOUT = 0.2
OPTIMIZER = 'adam'


def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    #for item in line1:
    ax.plot(line1, linewidth=lw)
    #for item in line2:
    ax.plot(line2, linewidth=lw)
    ax.set_ylabel('price [ETH]', fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.show()

def train_the_model(epochs, batch_size, window_len, input_columns, lstm_neurons, loss, dropout, optimizer, X_train, y_train, graph=True, summary=True):
    print('\n')
    model_folder = input(colored('Describe the model: ', 'green'))
    path = os.path.join('trained_models', model_folder)
    from learning import build_lstm_model, metrics
    print(colored('Building model...', 'green'))
    model = build_lstm_model(window_len=window_len, input_columns=input_columns, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,optimizer=optimizer)
    if summary:
        print(model.summary())
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True, callbacks=metrics)

    model_json = model.to_json()
    with open("eth_pred_model.json","w") as json_file:
        json_file.write(model_json)
    
    np.save('history.npy',history.history)

    os.mkdir(path) if not isdir(path) else [shutil.rmtree(path), os.mkdir(path)]
    os.rename("history.npy", os.path.join(path, "history.npy"))
    os.rename("eth_pred_model.json", os.path.join(path, "eth_pred_model.json"))
    os.rename("eth_pred_model_weights.hdf5", os.path.join(path, "eth_pred_model_weights.hdf5"))
    
    if graph:
        preds = model.predict(X_test)
        line_plot(y_test, preds, 'training', 'test', title='ETH price prediction')
    return model

def test_the_model(model_folder, max_size, graph=True):
    from learning import load_model
    print(colored(f'Loading model {model_folder}', 'green'))
    model = load_model(model_folder, OPTIMIZER, LOSS)
    max_size = len(X_test) if max_size > len(X_test) else max_size
    print(colored(f'Running tests on {max_size} samples', 'green'))
    preds = model.predict(X_test[:max_size])

    print(colored(preds, (30, 80, 80)))
    
    if graph:
        line_plot(y_test[:max_size], preds, 'training', 'test', title='ETH price prediction')
    return preds
  
if __name__=='__main__':
    dataframe = DataFrame(DATA_FILENAME)

    _, _, X_train, X_test, y_train, y_test = dataframe.prepare_data(window_len=WINDOW_LEN, zero_base=ZERO_BASE, test_size=TEST_SIZE)

    '''Visualizing the dataset'''
    print(X_train[:3])
    print('--')
    print(y_train[:3])

    model = train_the_model(EPOCHS, BATCH_SIZE, WINDOW_LEN, INPUT_COLUMNS, GRU_NEURONS, LOSS, DROPOUT, OPTIMIZER, X_train, y_train, graph=True) 

    # test_the_model('dense badboi v4', 200)