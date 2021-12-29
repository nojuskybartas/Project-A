import pickle
import pandas as pd
import logging


def extract_window_data(data, window_len):
    datas = [data.iloc[i:i+window_len].values for i in range(len(data) - window_len)]
    indices = [data.index[i+window_len] for i in range(len(data) - window_len)]
    return pd.DataFrame(datas, index=indices)


def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df['Close'].iloc[:split_row]
    test_data = df['Close'].iloc[split_row:]
    return train_data, test_data


class DataClass:
    def __init__(self, data_filename, window_len, test_size=0.2, debug=False):
        with open(data_filename, 'rb') as f:
            self.df = pd.DataFrame(pickle.load(f))

        d = list(self.df['Close'])
        self.df['dayChange'] = [0]+[d[i+1] - v for i, v in enumerate(d) if v is not d[-1]]
        self.df['dayPercentChange'] = [0]+[(100 * d[i+1] / v) - 100 for i, v in enumerate(d) if v is not d[-1]]

        self.train_data, self.test_data = train_test_split(self.df, test_size=test_size)

        self.y_train = self.train_data.iloc[window_len:]
        self.x_train = extract_window_data(self.train_data, window_len)

        self.y_test = self.test_data.iloc[window_len:]
        self.x_test = extract_window_data(self.test_data, window_len)

        logging.info(f"Train Data: {self.y_train.index[0]} ------ {self.y_train.index[-1]}")
        logging.info(f"Test Data: {self.y_test.index[0]} ------ {self.y_test.index[-1]}")

        assert(len(self.x_train) == len(self.y_train) and len(self.x_test) == len(self.y_test))

        if debug:
            print([element.values.shape + '\n' for element in
                   [self.train_data, self.test_data, self.x_train, self.x_test, self.y_train, self.y_test]])
