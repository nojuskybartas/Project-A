from dataclasses import dataclass
from os.path import isfile

# let's make a settings object for each model we use, like this there won't be any confusion about what settings
# to use when using it later, for example when testing with an already built model


@dataclass
class ModelSettings:
    MODEL_FOLDER: str
    DATA_FILENAME: str = 'data/ETH-USD_2016-01-01_UTC2021-11-07_UTC.data'
    DATA_FILENAME_TEST: str = 'data/ETH-USD_2021-11-07_UTC2021-11-24_UTC.data'
    WINDOW_LEN: int = 14
    INPUT_COLUMNS: int = 1  # columns in the dataframe used for training
    TEST_SIZE: float = 0.1
    ZERO_BASE: bool = False
    GRU_NEURONS: int = 3000
    EPOCHS: int = 250  # big number, because we have earlystopping
    BATCH_SIZE: int = 32
    LOSS: str = 'mse'
    DROPOUT: float = 0.2
    OPTIMIZER: str = 'adam'

    def __post_init__(self):
        assert(isfile(self.DATA_FILENAME) and isfile(self.DATA_FILENAME_TEST))

    def identifier(self):
        return (f"{self.DATA_FILENAME[5:-5]}{self.DATA_FILENAME_TEST[5:-5]}{self.WINDOW_LEN}{self.INPUT_COLUMNS}"
                f"{self.TEST_SIZE}{self.ZERO_BASE}{self.GRU_NEURONS}{self.EPOCHS}{self.BATCH_SIZE}"
                f"{self.LOSS}{self.DROPOUT}{self.OPTIMIZER}")


gru_bigboi3_settings = ModelSettings(MODEL_FOLDER='dense badboi v3', WINDOW_LEN=14, INPUT_COLUMNS=1, TEST_SIZE=0.1,
                                     ZERO_BASE=False, GRU_NEURONS=3000, EPOCHS=250, BATCH_SIZE=32, LOSS='mse',
                                     DROPOUT=0.2, OPTIMIZER='adam')
