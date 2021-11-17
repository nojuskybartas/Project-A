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

DATA_FILENAME = 'data/ETH-USD_2016-01-01_UTC2021-11-07_UTC.data'
assert(isfile(DATA_FILENAME))
np.random.seed(42)
WINDOW_LEN = 7
TEST_SIZE = 0.05
ZERO_BASE = False
LSTM_NEURONS = 1000
EPOCHS = 100
BATCH_SIZE = 64
LOSS = 'mse'
DROPOUT = 0.2
OPTIMIZER = 'adam'

def prep_df(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)['Close']
    return df

def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def extract_window_data(df, window_len, zero_base):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    
    return np.array(window_data)

def prepare_data(df, window_len, zero_base, test_size=0.2, debug=False):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[window_len:].values
    y_test = test_data[window_len:].values
    if zero_base:
        y_train = y_train / train_data[:-window_len].values - 1
        y_test = y_test / test_data[:-window_len].values - 1

    if debug:
        for data in [train_data, test_data, X_train, X_test, y_train, y_test]:
            print(data.shape, '\n')
    return train_data, test_data, X_train, X_test, y_train, y_test

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    #for item in line1:
    ax.plot(line1, linewidth=lw)
    #for item in line2:
    ax.plot(line2, linewidth=lw)
    ax.set_ylabel('price [ETH]', fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.show()

def train_the_model(epochs, batch_size, window_len, lstm_neurons, loss, dropout, optimizer, X_train, y_train, graph=True):
    print('\n')
    model_folder = input(colored('Describe the model: ', 'green'))
    path = os.path.join('trained_models', model_folder)
    from learning import build_lstm_model, metrics
    model = build_lstm_model(window_len=window_len, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,optimizer=optimizer)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True, callbacks=metrics)

    model_json = model.to_json()
    with open("eth_pred_model.json","w") as json_file:
        json_file.write(model_json)

    os.mkdir(path) if not isdir(path) else [shutil.rmtree(path), os.mkdir(path)]
    os.rename("eth_pred_model.json", os.path.join(path, "eth_pred_model.json"))
    os.rename("eth_pred_model_weights.hdf5", os.path.join(path, "eth_pred_model_weights.hdf5"))
    
    if graph:
        preds = model.predict(X_test)
        line_plot(y_test, preds, 'training', 'test', title='ETH price prediction')
    return model

def test_the_model(model_folder, max_size):
    from learning import load_model
    model = load_model(model_folder, OPTIMIZER, LOSS)
    max_size = len(X_test) if max_size > len(X_test) else max_size
    preds = model.predict(X_test[:max_size])
    print(preds)
    return preds
  
df = prep_df(DATA_FILENAME)

_, _, X_train, X_test, y_train, y_test = prepare_data(df, window_len=WINDOW_LEN, zero_base=ZERO_BASE, test_size=TEST_SIZE)

model = train_the_model(EPOCHS, BATCH_SIZE, WINDOW_LEN, LSTM_NEURONS, LOSS, DROPOUT, OPTIMIZER, X_train, y_train, graph=False) 

test_the_model('100 epochs 1000 neurons', 10)