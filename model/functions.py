import os
import logging
from model.container import ModelContainer
import model.architecture as architecture
from model.utils import line_plot
import numpy as np
import pandas as pd

np.random.seed(42)


def train(model: ModelContainer, show_testgraph=True, model_summary=True):
    data = model.data
    config = model.config

    logging.info('Building model...')

    try:
        model = architecture.build(window_len=config.WINDOW_LEN, input_columns=config.INPUT_COLUMNS, output_size=1,
                                   neurons=config.GRU_NEURONS, dropout=config.DROPOUT, loss=config.LOSS,
                                   optimizer=config.OPTIMIZER)
        logging.info('Model successfully built!')
    except Exception as e:
        logging.info(f'Failed to build model: {e}')
        return

    if model_summary:
        logging.info(model.summary())

    logging.info(
        f'Starting training of {config.MODEL_FOLDER} for {config.EPOCHS} epochs with early-stopping patience: '
        f'{model.architecture.PATIENCE}')

    history = model.fit(data.x_train, data.y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=1,
                        shuffle=True, validation_split=0.1, callbacks=architecture.model_metrics)
    architecture.save(model, history, model)

    if show_testgraph:
        preds = model.predict(data.x_test)
        logging.info(preds)
        line_plot(data.y_test, 'training', preds, 'test', title=f'{config.CURRENCY} price prediction',
                  path=os.path.join('trained_models', config.MODEL_FOLDER, 'testGraph.png'))

    return model


def run_inference(model: ModelContainer, days_to_predict, predict_forward=False, plot=True):
    x_test = model.data.x_test
    window_len = model.config.WINDOW_LEN
    window = x_test[window_len:window_len+1].values
    last_window_data = window.copy()
    assert len(x_test) >= days_to_predict + window_len

    idx_pred = x_test.index[window_len:window_len+days_to_predict]
    idx_window_data = x_test.index[0:window_len]

    logging.info(f'Making a {days_to_predict} day prediction '
                 f'(Predict forward: {predict_forward}) from \n{idx_pred[0]} ----- {idx_pred[-1]}')
    logging.info(f'Using window \n{idx_window_data[0]} ----- {idx_window_data[-1]}')
    predictions = []

    for i in range(days_to_predict):
        prediction = model.model.predict(window)
        predictions.append(prediction.tolist()[0])
        print(f'Generating: {i*100//days_to_predict}%', end='\r')

        # add the prediction to the next window if we predict forward, else add the real data from y_test
        added = prediction if predict_forward else [[model.data.y_test.iloc[i + window_len]]]
        window = np.delete(window, 0, 1)
        window = np.append(window, added, axis=1)

    logging.info(f'\nDone!')

    predictions_with_dates = pd.DataFrame(predictions, index=idx_pred)

    if plot:
        given_data_with_dates = pd.DataFrame(last_window_data[0], index=idx_window_data)
        real_data = model.data.y_test[window_len:days_to_predict + window_len]

        # add the last window value to connect the graphs
        predictions_graph = pd.concat([given_data_with_dates.tail(1), predictions_with_dates])
        real_data = pd.concat([given_data_with_dates.tail(1), real_data])

        line_plot(
            predictions_graph, 'Prediction', given_data_with_dates, 'First Window', real_data, 'Real Data',
            title=f'Price Prediction of {days_to_predict} days -{"" if predict_forward else "non-"}Stacked Predictions',
            path=os.path.join('trained_models', model.config.MODEL_FOLDER, f'{days_to_predict} day prediction.png'))

    return predictions_with_dates
