import pandas as pd
import logging
from model.settings import ModelSettings
from cryptocomp.helpers import load_price_data


def extract_window_data(data, window_len):
    datas = [data.iloc[i:i+window_len].values for i in range(len(data) - window_len)]
    indices = [data.index[i+window_len] for i in range(len(data) - window_len)]
    return pd.DataFrame(datas, index=indices)


def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df['Close'].iloc[:split_row]
    test_data = df['Close'].iloc[split_row:]
    return train_data, test_data


def augment(df):
    d = list(df['Close'])
    df['dayChange'] = [0]+[d[i+1] - v for i, v in enumerate(d) if v is not d[-1]]
    df['dayPercentChange'] = [0]+[(100 * d[i+1] / v) - 100 for i, v in enumerate(d) if v is not d[-1]]
    return df


class DataClass:
    def __init__(self, config: ModelSettings, debug=False):
        self.train_data = load_price_data(config.TRAIN_DATES[0], config.TRAIN_DATES[1], config.SYMBOL, config.TIMEFRAME)
        self.test_data = load_price_data(config.TEST_DATES[0], config.TEST_DATES[1], config.SYMBOL, config.TIMEFRAME)

        self.y_train = self.train_data.iloc[config.WINDOW_LEN:]
        self.x_train = extract_window_data(self.train_data, config.WINDOW_LEN)

        self.y_test = self.test_data.iloc[config.WINDOW_LEN:]
        self.x_test = extract_window_data(self.test_data, config.WINDOW_LEN)

        logging.info(f"Train Data: {self.y_train.index[0]} ------ {self.y_train.index[-1]}")
        logging.info(f"Test Data: {self.y_test.index[0]} ------ {self.y_test.index[-1]}")

        assert(len(self.x_train) == len(self.y_train) and len(self.x_test) == len(self.y_test))

        if debug:
            print([element.values.shape + '\n' for element in
                   [self.train_data, self.test_data, self.x_train, self.x_test, self.y_train, self.y_test]])
