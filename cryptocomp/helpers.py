import logging
import pickle
from os.path import isfile, join
import vectorbt as vbt
from model.container import ModelContainer


def load_price_data(start, end, symbol, timeframe='daily'):
    # get data from yahoo finance (or locally if we requested it before)
    filename = join('data', f"{symbol}_{start.replace(' ', '_')}{end.replace(' ', '_')}_{timeframe}.data")

    if isfile(filename):
        logging.info("loading local price data")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        logging.info("requesting api for price data")
        # possibilities: cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        interval = "1h" if timeframe == 'hourly' else "1d"
        data = vbt.YFData.download(symbol, start=start, end=end, cols=['Close'], interval=interval).get('Close')
        if data.empty:
            logging.critical("Failed to download API data")
            return None
        with open(filename, 'wb+') as f:
            data.to_pickle(f)
        return data


def load_prediction_data(model: ModelContainer, max_size: int, predict_forward=False, plot=True):
    filename = join('data/predictions',
                    f"{model.config.MODEL_FOLDER}_{max_size}_{predict_forward}_{model.__hash__()}.prediction")

    if isfile(filename):
        logging.info("loading stored prediction")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        logging.info("building prediction")
        future_prediction = model.run_inference(max_size, predict_forward=predict_forward, plot=plot)
        logging.info("storing prediction")
        with open(filename, 'wb+') as f:
            future_prediction.to_pickle(f)
        return future_prediction
