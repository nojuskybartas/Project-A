from dataclasses import dataclass
from typing import Any
import pandas as pd
import vectorbt as vbt


@dataclass
class MinimumStrategy:
    price: Any
    entries: Any = None
    exits: Any = None
    portfolio: Any = None

    def __str__(self):
        returns = self.portfolio.total_return() if self.portfolio else "Not Available"
        return f"{self.__class__.__name__} - total return: {returns}\n"


@dataclass
class HoldingStrategy(MinimumStrategy):
    def __post_init__(self):
        self.portfolio = vbt.Portfolio.from_holding(self.price, init_cash=100)


@dataclass
class MaCrossoverStrategy(MinimumStrategy):
    def __post_init__(self):
        fast_ma = vbt.MA.run(self.price, 10, short_name='fast')
        slow_ma = vbt.MA.run(self.price, 20, short_name='slow')
        self.entries = fast_ma.ma_above(slow_ma, crossover=True)
        self.exits = fast_ma.ma_below(slow_ma, crossover=True)
        self.portfolio = vbt.Portfolio.from_signals(self.price, self.entries, self.exits, init_cash=100)


@dataclass
class RsiStrategy(MinimumStrategy):
    def __post_init__(self):
        self.portfolio = vbt.RSI.run(self.price)
        entries = self.portfolio.rsi_below(30, crossover=False, wait=1)
        exits = self.portfolio.rsi_above(90, crossover=False, wait=50)
        self.entries, self.exits = pd.DataFrame.vbt.signals.clean(entries, exits)
        self.portfolio = vbt.Portfolio.from_signals(self.price, self.entries, self.exits, init_cash=100)


@dataclass
class NeuralStrategy(MinimumStrategy):
    def __post_init__(self):
        self.portfolio = vbt.Portfolio.from_signals(self.price, self.entries, self.exits,
                                                    init_cash=100, direction='both', size=1)


def get_simple_entries_exits(prediction, price_data, clamp_buys=8, clamp_sells=6):
    entries = [False]
    exits = [False]
    bought_in = False
    counter_sells = 0
    counter_buys = 0

    for i in range(prediction.size - 1):
        pred_today = prediction.values[i]
        pred_tomorrow = prediction.values[i + 1]

        if pred_tomorrow > pred_today and not bought_in:
            if counter_buys < clamp_buys:
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
            if counter_sells < clamp_sells:
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

    return pd.Series(entries, index=price_data.index), pd.Series(exits, index=price_data.index)
