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

def x_frequencies(network, x_col):
    connections = network.all_connections.copy()
    connections = connections.dropna()
    connections = connections[(connections['timestamp'] >= 1000) & (connections['timestamp'] <= 7000)]
    video_counts = connections[x_col].value_counts()
    ax = video_counts.plot(kind='hist', bins=50)
    ax.figure.show()

    return ax


def video_saturation(network, n_times=10):
    # plot number of videos that has not been seen 10 times yet over time
    time_steps = network.all_connections['timestamp'].max() + 1
    video_n = len(network.all_connections['video_id'].unique())
    # keep only the first 10 connections per video
    connections = network.all_connections.copy()
    connections = connections.dropna()
    connections = connections.sort_values('timestamp')
    connections['connection_count'] = connections.groupby('video_id').cumcount() + 1
    min_con = connections[connections['connection_count'] == n_times]
    cum = min_con.groupby('timestamp').size().cumsum()
    line = plt.plot(cum.index, cum.values)[0]
    line.figure.show()
    return line


def video_lifetime(network):
    # plot the timestep difference between the first time and the last time
    # a video was seen

    connections = network.all_connections.copy()
    connections = connections.dropna()
    connections = connections[connections['timestamp'] >= 1000]
    connections = connections[connections['timestamp'] <= 7000]
    video_first = connections.groupby('video_id')['timestamp'].min()
    video_last = connections.groupby('video_id')['timestamp'].max()
    video_life = video_last - video_first
    ax = video_life.plot(kind='hist', bins=50)
    ax.figure.show()
    return ax



if __name__ == "__main__":
    pd.set_option('display.max_columns',  None)
    try:
        df = pd.read_csv(os.path.join('..', 'data', 'raw', 'big_matrix.csv'))
        features = pd.read_csv(os.path.join('..', 'data', 'raw', 'user_features.csv'))
    except FileNotFoundError:
        df = pd.read_csv(os.path.join('data', 'raw', 'big_matrix.csv'))
        features = pd.read_csv(os.path.join('data', 'raw', 'user_features.csv'))
    network = BiNetwork(df, features)
    # ax = graph_timestamp_frequencies(network)
    # ax.fig.show()

    # line = video_saturation(network, n_times=100)

    # ax = video_lifetime(network)

    ax = x_frequencies(network, 'user_id')
