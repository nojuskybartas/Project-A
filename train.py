import os
import coloredlogs
os.environ['COLOREDLOGS_LEVEL_STYLES'] = 'info=blue'  # noqa: E402
coloredlogs.install()  # noqa: E402
from model.settings import gru_bigboi3
from model.container import ModelContainer


bigboi = ModelContainer(gru_bigboi3)
# bigboi.train()
bigboi.run_inference(100)
