from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib


@dataclass
class ModelSettings:
    MODEL_FOLDER: str
    SYMBOL: str
    TIMEFRAME: str
    WINDOW_LEN: int
    TRAIN_DATES: tuple[datetime, datetime]
    TEST_DATES: tuple[datetime, datetime]
    INPUT_COLUMNS: int = 1  # columns in the dataframe used for training
    GRU_NEURONS: int = 3000
    EPOCHS: int = 250  # big number, because we have earlystopping
    BATCH_SIZE: int = 32
    LOSS: str = 'mse'
    DROPOUT: float = 0.2
    OPTIMIZER: str = 'adam'

    def __hash__(self):
        return hashlib.md5(str(self).encode('utf-8')).hexdigest()


gru_bigboi3 = ModelSettings(MODEL_FOLDER='dense badboi v3', SYMBOL='ETH-USD', TIMEFRAME='1d',
                            TRAIN_DATES=(datetime(2016, 1, 1, tzinfo=timezone.utc),
                                         datetime(2021, 4, 8, tzinfo=timezone.utc)),
                            TEST_DATES=(datetime(2021, 4, 9, tzinfo=timezone.utc),
                                        datetime(2022, 1, 8, tzinfo=timezone.utc)),
                            WINDOW_LEN=14, INPUT_COLUMNS=1, GRU_NEURONS=3000, EPOCHS=250, BATCH_SIZE=32, LOSS='mse',
                            DROPOUT=0.2, OPTIMIZER='adam')
