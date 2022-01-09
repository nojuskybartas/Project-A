import logging
import pickle
from os.path import isfile, join
import vectorbt as vbt


def load_price_data(start, end, symbol, timeframe):
    assert timeframe in ('1h', '1d')
    # get data from yahoo finance (or locally if we requested it before)
    filename = join('data/price_data', f"{symbol}_{start.year}_{start.month}_{start.day}_{start.tzinfo}_"
                                       f"{end.year}_{end.month}_{end.day}_{end.tzinfo}_{timeframe}.data")

    if isfile(filename):
        logging.info("loading local price data")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        logging.info("requesting api for price data")
        # possibilities: cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = vbt.YFData.download(symbol, start=start, end=end, cols=['Close'], interval=timeframe).get('Close')
        if data.empty:
            logging.critical("Failed to download API data")
            return None
        with open(filename, 'wb+') as f:
            data.to_pickle(f)
        return data
