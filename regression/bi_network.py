import pandas as pd
import os


class BiNetwork:

    def __init__(self, interaction_df, user_features, filter_na_users=True):
        # drop all besides one hot encoded features
        self.user_features = user_features.drop(user_features.columns[list(range(1, 13))], axis=1)
        # self.user_features = user_features.drop(user_features.columns[list([1,2,5,6,8,9,11])], axis=1)
        self.interaction_df = interaction_df
        if filter_na_users:
            users_with_na = user_features[user_features.isna().any(axis=1)]['user_id']
            self.user_features = self.user_features[~self.user_features['user_id'].isin(users_with_na)].reset_index(drop=True)
            self.interaction_df = interaction_df[~interaction_df['user_id'].isin(users_with_na)].reset_index(drop=True)
        interaction_df['timestamp'] = (interaction_df['timestamp'] / 100).round(0)
        interaction_df['timestamp'] = interaction_df['timestamp'] - interaction_df['timestamp'].min()
        timestamps = interaction_df['timestamp'].unique()
        timestamps.sort()
        series = range(len(timestamps))
        timestamp_mapping = dict(zip(timestamps, series))
        interaction_df['timestamp'] = interaction_df['timestamp'].map(timestamp_mapping)
        self.all_connections = self.interaction_df.pivot_table(index=['user_id', 'video_id'], values=['watch_ratio', 'timestamp'])
        # create full network of all possible user-video connections
        users = interaction_df['user_id'].unique()
        videos = interaction_df['video_id'].unique()
        full_network = pd.MultiIndex.from_product([users, videos], names=['user_id', 'video_id']).to_frame(index=False)
        full_network = full_network.merge(self.interaction_df[['user_id', 'video_id', 'timestamp', 'watch_ratio']],
                                          on=['user_id', 'video_id'], how='left')
        self.all_connections = full_network
        # set a compound network to the same shape as all but empty

    def connections_at_t(self, t):
        return self.all_connections[self.all_connections['timestamp'] == t]

    def compound_at_t(self, t):
        return self.all_connections[self.all_connections['timestamp'] < t]

    def possible_at_t(self, t):
        # all connections that can happen and did not happen yet
        return self.all_connections[(self.all_connections['timestamp'] >= t) | (self.all_connections['timestamp'].isna())]

    def video_user_features_t(self, t):
        compound_at_t = self.compound_at_t(t)
        merged = compound_at_t.reset_index(drop=True).merge(self.user_features, on='user_id', how='left')
        merged = merged.drop(columns=['user_id', 'timestamp', 'watch_ratio'])
        video_features = merged.groupby('video_id').mean().reset_index(drop=False)

        return video_features


if __name__ == "__main__":
    pd.set_option('display.max_columns',  None)
    df = pd.read_csv(os.path.join('..', 'data', 'raw', 'big_matrix.csv'))
    features = pd.read_csv(os.path.join('..', 'data', 'raw', 'user_features.csv'))
    print(features[features['user_id'] == 2783])
    network = BiNetwork(df, features)
    print(network.all_connections)
    print(network.all_connections[network.all_connections['timestamp'].isna()])
    for i in range(1000):
        print(network.connections_at_t(i))

