import logging
import pandas as pd
from model import test_the_model
from data_api.helpers import load_price_data
from data_api.strategies import HoldingStrategy, NeuralStrategy, MaCrossoverStrategy

logging.getLogger().setLevel(logging.INFO)
prediction, testing_data, data = test_the_model('dense badboi v3', 300, graph=False)

entries = [False]
exits = [False]

bought_in = False
counter_sells = 0
counter_buys = 0

for i in range(prediction.size - 1):
    pred_today = prediction.values[i]
    pred_tomorrow = prediction.values[i+1]

    if pred_tomorrow > pred_today and not bought_in:
        if counter_buys < 8:
            counter_buys += 1
            entries.append(False)
            exits.append(False)
            continue
        else:
            counter_buys = 0
            entries.append(True)
            exits.append(False)
            bought_in = True
    elif pred_tomorrow <= pred_today and bought_in:
        if counter_sells < 6:
            counter_sells += 1
            entries.append(False)
            exits.append(False)
            continue
        else:
            counter_sells = 0
            entries.append(False)
            exits.append(True)
            bought_in = False
    else:
        entries.append(False)
        exits.append(False)

price = load_price_data('2021-04-09 UTC', '2021-10-24 UTC', 'ETH-USD')
hold = HoldingStrategy(price)

entries = pd.Series(entries, index=price.index)
exits = pd.Series(exits, index=price.index)
neural = NeuralStrategy(price, entries, exits)
ma = MaCrossoverStrategy(price)
print(hold, ma, neural)
neural.portfolio.plot().show()
