import os
import pandas as pd
import matplotlib.ticker as ticker

from matplotlib import pyplot as plt
from regression.bi_network import BiNetwork


def graph_timestamp_frequencies(network):
    a = network.all_connections['timestamp'].value_counts().sort_index().plot(kind='bar')
    ticks = a.get_xticks()
    a.set_xticks(ticks / 1000)
    a.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    a.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    return a


if __name__ == "__main__":
    pd.set_option('display.max_columns',  None)
    try:
        df = pd.read_csv(os.path.join('..', 'data', 'raw', 'big_matrix.csv'))
        features = pd.read_csv(os.path.join('..', 'data', 'raw', 'user_features.csv'))
    except FileNotFoundError:
        df = pd.read_csv(os.path.join('data', 'raw', 'big_matrix.csv'))
        features = pd.read_csv(os.path.join('data', 'raw', 'user_features.csv'))
    network = BiNetwork(df, features)
    ax = graph_timestamp_frequencies(network)
    ax.fig.show()
