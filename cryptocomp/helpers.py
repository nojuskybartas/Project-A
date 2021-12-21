import logging
import pickle
from os.path import isfile, join
import vectorbt as vbt
from cryptocomp.strategies import MinimumStrategy
from model import ModelSettings


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


def interactive_chart_for_strategy(strategy: MinimumStrategy):
    fig = vbt.make_subplots(specs=[[{"secondary_y": True}]])
    fig = strategy.price.vbt.plot(trace_kwargs=dict(name='Price'), fig=fig)
    if strategy.entries is not None and strategy.exits is not None:
        fig = strategy.entries.vbt.signals.plot_as_entry_markers(strategy.price, fig=fig)
        fig = strategy.exits.vbt.signals.plot_as_exit_markers(strategy.price, fig=fig)
    fig.show()


def load_prediction_data(config: ModelSettings, max_size: int):
    filename = join('data/predictions', f"{config.MODEL_FOLDER}_{max_size}_{config.__hash__()}.prediction")

    if isfile(filename):
        logging.info("loading stored prediction")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        from model import test_the_model
        logging.info("building prediction")
        future_prediction, testing_data, data = test_the_model(config, max_size, graph=False)
        logging.info("storing prediction")
        with open(filename, 'wb+') as f:
            future_prediction.to_pickle(f)
            return future_prediction
