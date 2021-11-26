import pickle
import pandas as pd
from os.path import isfile
import os

file = os.path.join("ETH-USD_2016-01-01_UTC2021-11-07_UTC.data")
assert (isfile(file))


class df_prep_by_diff:
    def __init__(self, filename):
        self.filename = filename

    def prep_df(self):
        with open(self.filename, 'rb') as f:
            data = pickle.load(f)

        df = pd.DataFrame(data)['Close']
        return df


# function that shows changes by day in a numerical value
def change_by_day(frame):
    value_list = []
    value_change_list = []
    for idx, row in enumerate(frame):
        value_list.append(row)

    for i in range(len(value_list) - 1):
        value_change_list.append(value_list[i + 1] - value_list[i])

    return value_change_list


# function that shows changes by day as a percent
def change_percentage_by_day(frame):
    value_list = []
    value_change_percent_list = []
    for idx, row in enumerate(frame):
        value_list.append(row)

    for i in range(len(value_list) - 1):
        value_change_percent_list.append((value_list[i + 1] * 100) / value_list[i] - 100)

    return value_change_percent_list


# some tests (feel free to delete)

test_data = df_prep_by_diff(file)
data_frame = test_data.prep_df()

print(change_by_day(test_data.prep_df()))
print("---------------------------------------------------------------------")
print(change_percentage_by_day(test_data.prep_df()))
