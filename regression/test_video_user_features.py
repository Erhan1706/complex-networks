import pandas as pd
import os
from regression.bi_network import BiNetwork


def preprocess_for_test(interaction_df, user_features):
    user_features = user_features.drop(user_features.columns[list(range(1, 13))], axis=1)
    test_features = user_features[['user_id', 'onehot_feat1']]
    interaction_df['timestamp'] = (interaction_df['timestamp'] / 100).round(0)
    interaction_df['timestamp'] = interaction_df['timestamp'] - interaction_df['timestamp'].min()
    timestamps = interaction_df['timestamp'].unique()
    timestamps.sort()
    series = range(len(timestamps))
    timestamp_mapping = dict(zip(timestamps, series))
    interaction_df['timestamp'] = interaction_df['timestamp'].map(timestamp_mapping)
    return interaction_df, user_features


def test_video_user_features():
    pass


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    try:
        small_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'small_matrix.csv'))
        print('small matrix read in')
        big_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'big_matrix.csv'))
        print('big matrix read in')
        features = pd.read_csv(os.path.join('..', 'data', 'raw', 'user_features.csv'))
        print('data read in')
    except:
        small_matrix = pd.read_csv(os.path.join('data', 'raw', 'small_matrix.csv'))
        print('small matrix read in')
        big_matrix = pd.read_csv(os.path.join('data', 'raw', 'big_matrix.csv'))
        print('big matrix read in')
        features = pd.read_csv(os.path.join('data', 'raw', 'user_features.csv'))
        print('data read in')
