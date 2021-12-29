import os
os.environ['COLOREDLOGS_LEVEL_STYLES'] = 'info=blue'  # noqa: E402
import logging
import coloredlogs
coloredlogs.install()  # noqa: E402
from cryptocomp.helpers import load_price_data, load_prediction_data
from cryptocomp.strategies import HoldingStrategy, NeuralStrategy, MaCrossoverStrategy, get_simple_entries_exits
from model.settings import gru_bigboi3
from model.container import ModelContainer
from datetime import datetime, timedelta, timezone

bigboi = ModelContainer(gru_bigboi3)

days_to_predict = 120
start = datetime(2021, 4, 9, tzinfo=timezone.utc)
end = start + timedelta(days=days_to_predict-1)

prediction = load_prediction_data(bigboi, days_to_predict, predict_forward=False)
price = load_price_data(f'{start:%Y-%m-%d UTC}', f'{end:%Y-%m-%d UTC}', 'ETH-USD', timeframe='daily')

entries, exits = get_simple_entries_exits(prediction, price)

logging.info("generating strategies")
hold = HoldingStrategy(price)
ma = MaCrossoverStrategy(price)
neural = NeuralStrategy(price, entries, exits)
logging.info("finished processing")

print(hold, ma, neural)
neural.portfolio.plot().show()
