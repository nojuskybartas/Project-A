from dataclasses import dataclass
from typing import Any
import pandas as pd
import vectorbt as vbt


@dataclass
class MinimumStrategy:
    price: Any
    portfolio: Any = None
    entries: Any = None
    exits: Any = None

    def __str__(self):
        returns = self.portfolio.total_return() if self.portfolio else "Not Available"
        return f"{self.__class__.__name__} - total return: {returns}\n"


@dataclass
class HoldingStrategy(MinimumStrategy):
    def __post_init__(self):
        self.portfolio = vbt.Portfolio.from_holding(self.price)


@dataclass
class MaCrossoverStrategy(MinimumStrategy):
    def __post_init__(self):
        fast_ma = vbt.MA.run(self.price, 10, short_name='fast')
        slow_ma = vbt.MA.run(self.price, 20, short_name='slow')
        self.entries = fast_ma.ma_above(slow_ma, crossover=True)
        self.exits = fast_ma.ma_below(slow_ma, crossover=True)
        self.portfolio = vbt.Portfolio.from_signals(self.price, self.entries, self.exits)


@dataclass
class RsiStrategy(MinimumStrategy):
    def __post_init__(self):
        # vbt.RSI.run(self.price).plot(levels=(40, 60)).show()
        self.portfolio = vbt.RSI.run(self.price)
        entries = self.portfolio.rsi_below(30, crossover=False, wait=1)
        exits = self.portfolio.rsi_above(90, crossover=False, wait=50)
        self.entries, self.exits = pd.DataFrame.vbt.signals.clean(entries, exits)
        self.portfolio = vbt.Portfolio.from_signals(self.price, self.entries, self.exits)