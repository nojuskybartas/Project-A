from os.path import isfile
from model.settings import ModelSettings


class ModelContainer:
    def __init__(self, config: ModelSettings):
        assert(isfile(config.DATA_FILENAME))
        self.config = config
        self._data = None
        self._model = None

    @property
    def data(self):
        if self._data is None:
            from model.dataclass import DataClass
            self._data = DataClass(self.config.DATA_FILENAME, self.config.WINDOW_LEN, self.config.TEST_SIZE)
        return self._data

    @property
    def model(self):
        if self._model is None:
            from model import architecture
            self._model = architecture.load(self.config.MODEL_FOLDER, self.config.OPTIMIZER, self.config.LOSS)
        return self._model

    def run_inference(self, days_to_predict, predict_forward=False, plot=True):
        from model.functions import run_inference
        return run_inference(self, days_to_predict, predict_forward, plot)

    def train(self, show_testgraph=True, model_summary=True):
        from model.functions import train
        return train(self, show_testgraph, model_summary)

    def __hash__(self):
        return self.config.__hash__()
