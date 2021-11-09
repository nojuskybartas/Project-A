import logging
import pickle
from os.path import isfile
import vectorbt as vbt
from strategies import MinimumStrategy


def load_price_data(start, end, symbol):
    # get data from yahoo finance (or locally if we requested it before)
    filename = f"../data/{symbol}_{start.replace(' ', '_')}{end.replace(' ', '_')}.data"

    if isfile(filename):
        logging.info("loading local data")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        logging.info("requesting api")
        # possibilities: cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = vbt.YFData.download(symbol, start=start, end=end, cols=['Close']).get('Close')
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
