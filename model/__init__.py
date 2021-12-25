import os
import logging
import shutil
import numpy as np
from os.path import isdir
from matplotlib import pyplot as plt
import pandas as pd
from termcolor import colored

from model.settings import ModelSettings, gru_bigboi3_settings
from model.dataclass import DataClass

np.random.seed(42)
logging.getLogger().setLevel(logging.INFO)


def line_plot(line1, line2, label1=None, label2=None, title='', lw=2, path=None):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    plt.legend(loc="upper left")
    ax.set_ylabel('price [ETH]', fontsize=14)
    ax.set_title(title, fontsize=16)
    if path is not None:
        plt.savefig(path)
        logging.info(colored(f'Plot successfully saved to {path}', 'green'))
    plt.show()


def save(model, history, config):
    model_json = model.to_json()
    with open(f"{config.CURRENCY}_pred_model.json", "w") as json_file:
        json_file.write(model_json)
    
    if history is not None:
        np.save('history.npy', history.history)

    path = os.path.join('trained_models', config.MODEL_FOLDER)
    os.mkdir(path) if not isdir(path) else [logging.info(colored(f'THis model version already exists', 'green')), shutil.rmtree(path), os.mkdir(path)]
    os.rename("history.npy", os.path.join(path, "history.npy"))
    os.rename(f"{config.CURRENCY}_pred_model.json", os.path.join(path, f"{config.CURRENCY}_pred_model.json"))
    os.rename("pred_model_weights.hdf5", os.path.join(path, f"{config.CURRENCY}_pred_model_weights.hdf5"))
    logging.info(colored(f'Trained model successfully saved to {path}', 'green'))


def train(config: ModelSettings, showTestGraph=True, modelSummary=True):
    from model.architecture import build_model, model_metrics
    from model.dataclass import DataClass 
    from model.settings import PATIENCE

    data = DataClass(config)

    logging.info(colored('Building model...', 'green'))
    try:
        model = build_model(window_len=config.WINDOW_LEN, input_columns=config.INPUT_COLUMNS, output_size=1,
                               neurons=config.GRU_NEURONS, dropout=config.DROPOUT, loss=config.LOSS, optimizer=config.OPTIMIZER)
        logging.info(colored('Model successfully built!', 'green'))
    except:
        logging.info(colored('Failed to build model', 'red'))
        return

    if modelSummary:
        logging.info(model.summary())

    logging.info(colored(f'Starting training of {config.MODEL_FOLDER} for {config.EPOCHS} epochs with early-stopping patience: {PATIENCE}', 'green'))
    history = model.fit(data.x_train, data.y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=1,
                        shuffle=True, validation_split=0.1, callbacks=model_metrics)

    save(model, history, config)
    
    if showTestGraph:
        preds = model.predict(data.x_test)
        logging.info(preds)
        line_plot(data.y_test, preds, 'training', 'test', title=f'{config.CURRENCY} price prediction', path=os.path.join('trained_models', config.MODEL_FOLDER, 'testGraph.png'))
    
    return model


def test(config: ModelSettings, size, graph=True):
    from model.architecture import load_model

    data = DataClass(config)

    logging.info(colored(f'Loading model {config.MODEL_FOLDER}...', 'green'))
    try:
        model = load_model(config.MODEL_FOLDER, config.OPTIMIZER, config.LOSS)
        logging.info(colored(f'Successfully loaded the model', 'green'))
    except:
        logging.info(colored(f'Failed to load the model', 'red'))
    size = len(data.x_test) if size > len(data.x_test) else size
    
    logging.info(colored(f'Running tests on {size} samples', 'green'))
    preds = model.predict(data.x_test[:size])

    # TODO: some sort of accuracy calculation

    predictions_with_dates = pd.DataFrame(data=preds, index=data.test_data.index[:size])
    y_test_with_dates = pd.DataFrame(data=data.y_test[:size], index=data.test_data.index[:size])

    if graph:
        line_plot(y_test_with_dates, predictions_with_dates, 'test_data', 'prediction', title=f'{config.CURRENCY} price prediction')
    return predictions_with_dates, y_test_with_dates, data
