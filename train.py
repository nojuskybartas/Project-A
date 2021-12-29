import os
from model.settings import gru_bigboi3
from model.container import ModelContainer
bigboi = ModelContainer(gru_bigboi3)
# bigboi.train()
bigboi.run_inference(100)
