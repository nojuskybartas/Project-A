import pandas as pd
import logging
from matplotlib import pyplot as plt
from os.path import join


def line_plot(line1, label1=None, line2=None, label2=None, line3=None, label3=None, title='', lw=2, path=None):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    if line2 is not None:
        ax.plot(line2, label=label2, linewidth=lw)
    if line3 is not None:
        ax.plot(line3, label=label3, linewidth=lw)
    plt.legend(loc="upper left")
    ax.set_ylabel('price [ETH]', fontsize=14)
    ax.set_xlabel('days', fontsize=14)
    ax.set_title(title, fontsize=16)
    if path is not None:
        plt.savefig(path)
        logging.info(f'Plot successfully saved to {path}')
    plt.show()


def inference_plot(model, predictions):
    meta = predictions.attrs
    given_data_with_dates = pd.DataFrame(meta['last_window_data'], index=meta['idx_window_data'])
    real_data = model.data.y_test[meta['window_len']:meta["days_to_predict"] + meta['window_len']]

    # add the last window value to connect the graphs
    predictions_graph = pd.concat([given_data_with_dates.tail(1), predictions])
    real_data = pd.concat([given_data_with_dates.tail(1), real_data])

    line_plot(
        predictions_graph, 'Prediction', given_data_with_dates, 'First Window', real_data, 'Real Data',
        title=f'Price Prediction {"" if not meta["predict_forward"] else "Stacked Predictions Only"}'
              f'\n {predictions.index[0].date()} to {predictions.index[-1].date()}',
        path=join('trained_models', model.config.MODEL_FOLDER,
                  f'{meta["days_to_predict"] + meta["steps_into_future"]} day prediction.png'))
