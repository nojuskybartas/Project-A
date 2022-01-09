import os
os.environ['COLOREDLOGS_LEVEL_STYLES'] = 'info=blue'  # noqa: E402
import coloredlogs
coloredlogs.install()  # noqa: E402
from model.settings import gru_bigboi3
from model.model import Model
from datetime import datetime, timezone

bigboi = Model(gru_bigboi3)
bigboi.config.TEST_DATES = (datetime(2021, 12, 1, tzinfo=timezone.utc),
                            datetime.now(tz=timezone.utc).replace(minute=0, hour=0, second=0, microsecond=0))
bigboi.predict(steps_into_future=8)

# from cryptocomp.strategies import simulate_strategy
# simulation_results = simulate_strategy(bigboi, prediction_length=80, folds=1, predict_forward=False, plot=False)
