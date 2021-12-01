import pickle
import pandas as pd
from os.path import isfile
import os
import numpy as np

file = os.path.join("data","ETH-USD_2016-01-01_UTC2021-11-07_UTC.data")
assert (isfile(file))


class DataFrame:
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'rb') as f:
            data = pickle.load(f)
        self.value_list = list(pd.DataFrame(data)['Close'])
        self.df = pd.DataFrame(data)
        # drop the first row to keep the length of the day-change arrays
        self.df = self.df.iloc[1: , :]
        # 
        self.augment_data()
        # self.df = self.df.drop(self.df.columns[[0, 0, 0]], axis=0)
        # self.df = pd.DataFrame(data)['Close']

    def augment_data(self):
        self.df['dayChange'] = self.change_by_day()
        self.df['dayPercentChange'] = self.change_percentage_by_day()

    def train_test_split(self, test_size=0.2):
        split_row = len(self.df) - int(test_size * len(self.df))
        self.train_data = self.df.iloc[:split_row]
        self.test_data = self.df.iloc[split_row:]
        return self.train_data, self.test_data

    def normalise_zero_base(self):
        return self.df / self.df.iloc[0] - 1

    def extract_window_data(self, df, window_len, zero_base):
        window_data = []
        for idx in range(len(df) - window_len):
            tmp = df[idx: (idx + window_len)].copy()
            if zero_base:
                tmp = self.normalise_zero_base(tmp)
            window_data.append(tmp.values)
        
        return np.array(window_data)        

    def prepare_data(self, window_len, zero_base, test_size=0.2, debug=False):
        train_data, test_data = self.train_test_split(test_size=test_size)
        X_train = self.extract_window_data(train_data, window_len, zero_base)
        X_test = self.extract_window_data(test_data, window_len, zero_base)
        y_train = train_data[window_len:]['Close'].values
        y_test = test_data[window_len:]['Close'].values
        # print(X_train.shape, y_train.shape)
        assert(len(X_train) == len(y_train))
        
        
        if zero_base:
            y_train = y_train / train_data[:-window_len].values - 1
            y_test = y_test / test_data[:-window_len].values - 1

        if debug:
            for data in [train_data, test_data, X_train, X_test, y_train, y_test]:
                print(data.shape, '\n')
        return train_data, test_data, X_train, X_test, y_train, y_test

    # function that shows changes by day in a numerical value
    def change_by_day(self):
        self.value_change_list = []

        for i in range(len(self.value_list) - 1):
            self.value_change_list.append(self.value_list[i + 1] - self.value_list[i])

        return self.value_change_list

    # function that shows changes by day as a percent
    def change_percentage_by_day(self):
        self.value_change_percent_list = []

        for i in range(len(self.value_list) - 1):
            self.value_change_percent_list.append((self.value_list[i + 1] * 100) / self.value_list[i] - 100)

        return self.value_change_percent_list


if __name__=='__main__':
    dataframe = DataFrame(file)

    print(dataframe.df.head())
    print(dataframe.df.tail())
    print("---------------------------------------------------------------------")
    print(dataframe.change_by_day()[:5])
    print("---------------------------------------------------------------------")
    print(dataframe.change_percentage_by_day()[:5])
