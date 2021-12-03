import os
import logging
import shutil
import numpy as np
import matplotlib.pyplot as plt
from os.path import isdir

import pandas as pd
from termcolor import colored

from model.modelsettings import ModelSettings, gru_bigboi3_settings
from model.dataframe import DataFrame

np.random.seed(42)
logging.getLogger().setLevel(logging.INFO)


class DataContainer:
    def __init__(self, config: ModelSettings, data_filename=None):
        self.config: ModelSettings = config
        if not data_filename:
            data_filename = config.DATA_FILENAME
        self.dataframe = DataFrame(data_filename)
        self.train_data, self.test_data, self.x_train, self.x_test, self.y_train, self.y_test = \
            self.dataframe.prepare_data(config.WINDOW_LEN, config.ZERO_BASE, config.TEST_SIZE)


def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    plt.legend(loc="upper left")
    ax.set_ylabel('price [ETH]', fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.show()


def train_the_model(epochs, batch_size, window_len, input_columns, lstm_neurons, loss, dropout, optimizer,
                    data=None, graph=True, summary=True):
    from model.learning import build_neural_model, model_metrics
    model_folder = input(colored('Describe the model: ', 'green'))
    path = os.path.join('trained_models', model_folder)

    logging.info('Building model...')
    model = build_neural_model(window_len=window_len, input_columns=input_columns, output_size=1,
                               neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)

    if summary:
        print(model.summary())
    history = model.fit(data.x_train, data.y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                        shuffle=True, callbacks=model_metrics)

    model_json = model.to_json()
    with open("eth_pred_model.json", "w") as json_file:
        json_file.write(model_json)
    
    np.save('history.npy', history.history)

    os.mkdir(path) if not isdir(path) else [shutil.rmtree(path), os.mkdir(path)]
    os.rename("history.npy", os.path.join(path, "history.npy"))
    os.rename("eth_pred_model.json", os.path.join(path, "eth_pred_model.json"))
    os.rename("eth_pred_model_weights.hdf5", os.path.join(path, "eth_pred_model_weights.hdf5"))
    
    if graph:
        preds = model.predict(data.x_test)
        line_plot(data.y_test, preds, 'training', 'test', title='ETH price prediction')
    return model


def test_the_model(model_folder, max_size, data: DataContainer = None, graph=True):
    from model.learning import load_model
    if not data:
        data: DataContainer = DataContainer(gru_bigboi3_settings)

    logging.info(colored(f'Loading model {model_folder}', 'green'))
    model = load_model(model_folder, data.config.OPTIMIZER, data.config.LOSS)
    max_size = len(data.x_test) if max_size > len(data.x_test) else max_size
    logging.info(colored(f'Running tests on {max_size} samples', 'green'))

    preds = model.predict(data.x_test[:max_size])
    predictions_with_dates = pd.DataFrame(data=preds, index=data.test_data.index[:max_size])
    y_test_with_dates = pd.DataFrame(data=data.y_test[:max_size], index=data.test_data.index[:max_size])

    if graph:
        line_plot(y_test_with_dates, predictions_with_dates, 'test_data', 'prediction',  title='ETH price prediction')
    return predictions_with_dates, y_test_with_dates, data


if __name__ == '__main__':
    m_data = DataContainer(gru_bigboi3_settings)

    '''Visualizing the dataset'''
    print(m_data.x_train[:3])
    print('--')
    print(m_data.y_train[:3])

    c = m_data.config
    the_model = train_the_model(c.EPOCHS, c.BATCH_SIZE, c.WINDOW_LEN, c.INPUT_COLUMNS,
                                c.GRU_NEURONS, c.LOSS, c.DROPOUT, c.OPTIMIZER, data=m_data, graph=True)
    # test_the_model('dense badboi v4', 200, data=m_data)
