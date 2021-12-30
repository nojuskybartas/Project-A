import os
os.environ['COLOREDLOGS_LEVEL_STYLES'] = 'info=blue'  # noqa: E402
import coloredlogs
coloredlogs.install()  # noqa: E402
from cryptocomp.strategies import simulate_strategy
from model.settings import gru_bigboi3
from model.container import ModelContainer

bigboi = ModelContainer(gru_bigboi3)
simulation_results = simulate_strategy(bigboi, prediction_length=80, folds=1, predict_forward=False, plot=False)
