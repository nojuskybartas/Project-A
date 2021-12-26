from matplotlib import pyplot as plt
from termcolor import colored
import logging
logging.getLogger().setLevel(logging.INFO)


def line_plot(line1, line2=None, label1=None, label2=None, title='', lw=2, path=None):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    if line2 is not None:
        ax.plot(line2, label=label2, linewidth=lw)
    plt.legend(loc="upper left")
    ax.set_ylabel('price [ETH]', fontsize=14)
    ax.set_xlabel('days', fontsize=14)
    ax.set_title(title, fontsize=16)
    if path is not None:
        plt.savefig(path)
        logging.info(colored(f'Plot successfully saved to {path}', 'green'))
    plt.show()
