import os
import logging
from termcolor import colored
from collections import deque
import numpy as np
import pandas as pd
import model.settings as settings
from model.settings import ModelSettings
import model.architecture as architecture
from model.utils import line_plot
from model.dataclass import DataClass
import numpy as np

np.random.seed(42)
logging.getLogger().setLevel(logging.INFO)


def train(config: ModelSettings, showTestGraph=True, modelSummary=True):

    data = DataClass(config)

    logging.info(colored('Building model...', 'green'))
    try:
        model = architecture.build(window_len=config.WINDOW_LEN, input_columns=config.INPUT_COLUMNS, output_size=1,
                                   neurons=config.GRU_NEURONS, dropout=config.DROPOUT, loss=config.LOSS, optimizer=config.OPTIMIZER)
        logging.info(colored('Model successfully built!', 'green'))
    except:
        logging.info(colored('Failed to build model', 'red'))
        return

    if modelSummary:
        logging.info(model.summary())

    logging.info(colored(
        f'Starting training of {config.MODEL_FOLDER} for {config.EPOCHS} epochs with early-stopping patience: {settings.PATIENCE}', 'green'))
    history = model.fit(data.x_train, data.y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=1,
                        shuffle=True, validation_split=0.1, callbacks=architecture.model_metrics)

    architecture.save(model, history, config)

    if showTestGraph:
        preds = model.predict(data.x_test)
        logging.info(preds)
        line_plot(data.y_test, preds, 'training', 'test', title=f'{config.CURRENCY} price prediction', path=os.path.join(
            'trained_models', config.MODEL_FOLDER, 'testGraph.png'))

    return model


def test(config: ModelSettings, size, graph=True):

    data = DataClass(config)

    model = architecture.load(
        config.MODEL_FOLDER, config.OPTIMIZER, config.LOSS)
    if model is None:
        return

    size = len(data.x_test) if size > len(data.x_test) else size

    logging.info(colored(f'Running tests on {size} samples', 'green'))
    preds = model.predict(data.x_test[:size])

    # TODO: some sort of accuracy calculation

    predictions_with_dates = pd.DataFrame(
        data=preds, index=data.test_data.index[:size])
    y_test_with_dates = pd.DataFrame(
        data=data.y_test[:size], index=data.test_data.index[:size])

    if graph:
        line_plot(y_test_with_dates, predictions_with_dates, 'test_data',
                  'prediction', title=f'{config.CURRENCY} price prediction')
    return predictions_with_dates, y_test_with_dates, data


def run_inference(model_folder, last_window_data: np.ndarray, days_to_predict=30, plot=True):

    model = architecture.load(model_folder)
    if model is None:
        return

    logging.info(
        colored(f'Making a {days_to_predict} day prediction', 'green'))
    predictions = []
    window = last_window_data
    for i in range(days_to_predict):
        prediction = model.predict(window)
        predictions.append(prediction.tolist()[0])
        window = np.delete(window, 13, 1)
        window = np.append(prediction, window, axis=1)
        print(f'Generating: {i*100/days_to_predict}%', end='\r')
    logging.info(colored(f'Done!', 'green'))
    
    for i in last_window_data[0]:
        predictions = np.append([[i]], predictions, axis=0)

    if plot:
        # our np array is built left-to-right (left-most datapoint is the newest), but for the graph, we need to reverse it
        last_window_data = list(reversed(last_window_data[0]))

        line_plot(predictions, last_window_data, 'Prediction', 'Given Data',
                f'Price Prediction of {days_to_predict} days', path=os.path.join('trained_models', model_folder, f'{days_to_predict} day prediction.png'))

    return predictions[len(last_window_data):]
