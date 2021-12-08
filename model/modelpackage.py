import pickle
import pandas as pd
from os.path import isfile
import numpy as np
from model import ModelSettings


class ModelPackagage:
    def __init__(self, config: ModelSettings):
        assert(isfile(config.DATA_FILENAME) and isfile(config.DATA_FILENAME_TEST))
        with open(config.DATA_FILENAME, 'rb') as f:
            self.df = pd.DataFrame(pickle.load(f))

        d = list(self.df['Close'])
        self.df['dayChange'] = [0]+[d[i+1] - v for i, v in enumerate(d) if v is not d[-1]]
        self.df['dayPercentChange'] = [0]+[(100 * d[i+1] / v) - 100 for i, v in enumerate(d) if v is not d[-1]]

        self.config = config
        self.train_data, self.test_data, self.x_train, self.x_test, self.y_train, self.y_test = \
            self.prepare_data(config.WINDOW_LEN, config.ZERO_BASE, config.TEST_SIZE)

    def train_test_split(self, test_size=0.2):
        split_row = len(self.df) - int(test_size * len(self.df))
        train_data = self.df['Close'].iloc[:split_row]
        test_data = self.df['Close'].iloc[split_row:]
        return train_data, test_data

    def normalise_zero_base(self):
        return self.df / self.df.iloc[0] - 1

    def extract_window_data(self, df, window_len, zero_base):
        window_data = []
        for idx in range(len(df) - window_len):
            tmp = df[idx: (idx + window_len)].copy()
            window_data.append(tmp.values if not zero_base else self.normalise_zero_base())
        return np.array(window_data)

    def prepare_data(self, window_len, zero_base, test_size=0.2, debug=False):
        train_data, test_data = self.train_test_split(test_size=test_size)
        x_train = self.extract_window_data(train_data, window_len, zero_base)
        x_test = self.extract_window_data(test_data, window_len, zero_base)
        y_train = train_data[window_len:].values
        y_test = test_data[window_len:].values

        assert(len(x_train) == len(y_train) and len(x_test) == len(y_test))
        if zero_base:
            y_train = y_train / train_data[:-window_len].values - 1
            y_test = y_test / test_data[:-window_len].values - 1
        if debug:
            print([element.shape + '\n' for element in [train_data, test_data, x_train, x_test, y_train, y_test]])

        return train_data, test_data, x_train, x_test, y_train, y_test
