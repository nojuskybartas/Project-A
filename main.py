from model import DataContainer, test_the_model, train_the_model
from model.modelsettings import gru_bigboi3_settings

m_data = DataContainer(gru_bigboi3_settings)

'''Visualizing the dataset'''
print(m_data.x_train[:3])
print('--')
print(m_data.y_train[:3])

c = m_data.config

# the_model = train_the_model(c.EPOCHS, c.BATCH_SIZE, c.WINDOW_LEN, c.INPUT_COLUMNS,
#                             c.GRU_NEURONS, c.LOSS, c.DROPOUT, c.OPTIMIZER, data=m_data, graph=True)

test_the_model('dense badboi v3', 200, data=m_data)
