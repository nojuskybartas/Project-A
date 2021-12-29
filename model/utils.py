from matplotlib import pyplot as plt
import logging


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
