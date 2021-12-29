from dataclasses import dataclass
import hashlib


@dataclass
class ModelSettings:
    MODEL_FOLDER: str
    CURRENCY: str
    TIMEFRAME: str
    DATA_FILENAME: str
    WINDOW_LEN: int
    INPUT_COLUMNS: int = 1  # columns in the dataframe used for training
    TEST_SIZE: float = 0.1
    GRU_NEURONS: int = 3000
    EPOCHS: int = 250  # big number, because we have earlystopping
    BATCH_SIZE: int = 32
    LOSS: str = 'mse'
    DROPOUT: float = 0.2
    OPTIMIZER: str = 'adam'

    def __hash__(self):
        return hashlib.md5(str(self).encode('utf-8')).hexdigest()


gru_bigboi3 = ModelSettings(MODEL_FOLDER='dense badboi v3', CURRENCY='eth', TIMEFRAME='1d',
                            DATA_FILENAME='data/ETH-USD_2016-01-01_UTC2021-11-07_UTC_daily.data',
                            WINDOW_LEN=14, INPUT_COLUMNS=1, TEST_SIZE=0.1,
                            GRU_NEURONS=3000, EPOCHS=250, BATCH_SIZE=32, LOSS='mse', DROPOUT=0.2, OPTIMIZER='adam')

bigboi_hourly = ModelSettings(MODEL_FOLDER='bigboi_hourly', CURRENCY='eth', TIMEFRAME='1h',
                              DATA_FILENAME='data/ETH-USD_2020-1-10_UTC2021-12-20_UTC_hourly.data',
                              WINDOW_LEN=120, INPUT_COLUMNS=1, TEST_SIZE=0.1,
                              GRU_NEURONS=3000, EPOCHS=250, BATCH_SIZE=16, LOSS='mse', DROPOUT=0.2, OPTIMIZER='adam')

bigboi_hourly_1 = ModelSettings(MODEL_FOLDER='bigboi_hourly_1', CURRENCY='eth', TIMEFRAME='1h',
                                DATA_FILENAME='data/ETH-USD_2020-1-10_UTC2021-12-20_UTC_hourly.data',
                                WINDOW_LEN=72, INPUT_COLUMNS=1, TEST_SIZE=0.1,
                                GRU_NEURONS=2048, EPOCHS=60, BATCH_SIZE=64, LOSS='mse', DROPOUT=0.2, OPTIMIZER='adam')
