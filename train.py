import os
from model.settings import gru_bigboi3
from model.model import Model
bigboi = Model(gru_bigboi3)
# bigboi.train()
bigboi.run_inference(100)
