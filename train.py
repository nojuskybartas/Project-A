import model

import model.settings as settings
from model.utils import line_plot


# model.test(settings.gru_bigboi3, 200)
# model.train(settings.bigboi)
dataset = model.DataClass(settings.gru_bigboi3)
last_window_data = dataset.x_test[177:178]


model.run_inference('dense badboi v3', last_window_data, 100)
