import logging
import pandas as pd
import pandas_ta
import yfinance
import matplotlib.pyplot as plt
from matplotlib import gridspec

import helpers
from strategies import HoldingStrategy, MaCrossoverStrategy, RsiStrategy

assert pd and pandas_ta and yfinance
logging.getLogger().setLevel(logging.INFO)

# get our data
price_data = helpers.load_price_data('2016-01-01 UTC', '2021-11-07 UTC', 'ETH-USD')

# now there are 2 paths: pandas, and vectorbt
# vectorbt actually uses pandas in the background, and is a more powerful library in many ways
# however it is a bit harder to use. at any point you can take a step back and work with pandas and the pandas dataframe
# vectorbt is based on arrays. for your own strategies you will want to work with signals (when to buy, when to sell)
# however regular signals won't work with vectorbt - we need to convert them to vectorbt's vector form
# you can investigate items such as ma_cross.entries (ma_cross = a MaCrossoverStrategy object like we do below)
# basically the vector is a field which contains all the present dates in the correct format
# and a boolean associated value for each date. for entries, all points of entry would have a True in the array
# and all other points a False. We might want to develop a function to convert from 'simple' signals to vectorbt signals
# or maybe it already exists and we need to do some research :)

# let's continue with the vectorbt path: setup 3 strategies
hold = HoldingStrategy(price_data)
ma_cross = MaCrossoverStrategy(price_data)
rsi = RsiStrategy(price_data)

# print the strategy objects to see the result. HoldingStrategy is the base. You might observe that
# MaCrossoverStrategy almost triples profits compared to simple holding (at least in this scenario)
print(hold, ma_cross, rsi)

# or show an interactive chart
helpers.interactive_chart_for_strategy(ma_cross)

#

#

# now let's try the pandas path
# get data and load it into pandas
df = pd.DataFrame(price_data)

# use pandas.ta package to generate the relative strength index and add it as a new column in the dataframe
df.ta.rsi(length=15, append=True)

# show the pandas dataframe with mathplotlib
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

plt.subplot(gs[0])
plt.plot(df['Close'])
plt.title("Price over Time")

plt.subplot(gs[1])
plt.plot(df['RSI_15'])
plt.title("RSI (15)")

plt.tight_layout()
plt.show()
