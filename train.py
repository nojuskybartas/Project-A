import model

bigboi = model.ModelSettings(MODEL_FOLDER='bigboi_hourly_1', CURRENCY='eth', TIMEFRAME='1h',
                            DATA_FILENAME='data/ETH-USD_2020-1-10_UTC2021-12-20_UTC_hourly.data',
                            DATA_FILENAME_TEST='data/ETH-USD_2020-1-10_UTC2021-12-20_UTC_hourly.data',
                            WINDOW_LEN=72, INPUT_COLUMNS=1, TEST_SIZE=0.9,
                            ZERO_BASE=False, GRU_NEURONS=2048, EPOCHS=250, BATCH_SIZE=128, LOSS='mse',
                            DROPOUT=0.2, OPTIMIZER='adam')

model.train(bigboi)