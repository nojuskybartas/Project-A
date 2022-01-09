import logging
import pickle
from typing import BinaryIO
from model.settings import ModelSettings
from os.path import isfile, join
from model.visualisation import line_plot, inference_plot


class Model:
    def __init__(self, config: ModelSettings):
        self.config = config
        self._data = None
        self._model = None
        self.predictions = None

    def __hash__(self):
        return self.config.__hash__()

    @property
    def data(self):
        if self._data is None:
            from model.dataclass import DataClass
            self._data = DataClass(self.config)
        return self._data

    @property
    def model(self):
        if self._model is None:
            from model import architecture
            self._model = architecture.load(self.config.MODEL_FOLDER, self.config.OPTIMIZER, self.config.LOSS)
        return self._model

    def train(self, show_testgraph=True, model_summary=True):
        import model.architecture as architecture
        data = self.data
        config = self.config

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
            line_plot(data.y_test, 'training', preds, 'test', title=f'{config.SYMBOL} price prediction',
                      path=join('trained_models', config.MODEL_FOLDER, 'testGraph.png'))

        return model

    def predict(self, prediction_length: int = -1, steps_into_future: int = 5, predict_forward=False, plot=True):
        filename = join('data/predictions', f"{self.config.MODEL_FOLDER}_{prediction_length}_{predict_forward}_"
                                            f"{steps_into_future}_{self.__hash__()}.prediction")

        if isfile(filename):
            logging.info("loading stored prediction")
            with open(filename, 'rb') as f:
                f: BinaryIO
                future_prediction = pickle.load(f)
        else:
            logging.info("building prediction")
            future_prediction = self.run_inference(prediction_length, steps_into_future, predict_forward)
            logging.info("storing prediction")
            with open(filename, 'wb+') as f:
                f: str
                future_prediction.to_pickle(f)

        if plot:
            inference_plot(self, future_prediction)

        self.predictions = future_prediction
        return future_prediction

    def run_inference(self, days_to_predict, steps_into_future=5, predict_forward=False):
        import numpy as np
        import pandas as pd
        np.random.seed(42)

        x_test = self.data.x_test
        window_len = self.config.WINDOW_LEN
        window = x_test[window_len:window_len+1].values
        last_window_data = window.copy()

        if days_to_predict == -1:
            days_to_predict = len(x_test) - window_len

        assert len(x_test) >= days_to_predict + window_len

        idx_pred = pd.date_range(x_test.index[window_len], periods=days_to_predict + steps_into_future, freq="D")
        idx_window_data = x_test.index[0:window_len]

        logging.info(f'Making a {days_to_predict} day prediction '
                     f'(Predict forward: {predict_forward}) from \n{idx_pred[0]} ----- {idx_pred[-1]}')
        logging.info(f'Using window \n{idx_window_data[0]} ----- {idx_window_data[-1]}')
        predictions = []

        for i in range(days_to_predict + steps_into_future):
            prediction = self.model.predict(window)
            predictions.append(prediction.tolist()[0])
            print(f'Generating: {i*100//(days_to_predict+steps_into_future)}%', end='\r')

            # add the prediction to the next window if we predict forward, else add the real data from y_test
            append_predict = predict_forward or (i + window_len) >= len(self.data.y_test)
            added = prediction if append_predict else [[self.data.y_test.iloc[i + window_len]]]

            window = np.delete(window, 0, 1)
            window = np.append(window, added, axis=1)

        print('\n')
        logging.info(f'Done!')

        predictions = pd.DataFrame(predictions, index=idx_pred)
        predictions.attrs = {'last_window_data': last_window_data[0],
                             'idx_window_data': idx_window_data,
                             'window_len': window_len,
                             'days_to_predict': days_to_predict,
                             'predict_forward': predict_forward,
                             'steps_into_future': steps_into_future}
        return predictions
