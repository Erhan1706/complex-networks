import pandas as pd
import os


class BiNetwork:

    def __init__(self, interaction_df, user_features):
        # drop all besides one hot encoded features
        self.user_features = user_features.drop(user_features.columns[list(range(1, 13))], axis=1)
        self.interaction_df = interaction_df
        interaction_df['timestamp'] = (interaction_df['timestamp'] / 100).round(0)
        interaction_df['timestamp'] = interaction_df['timestamp'] - interaction_df['timestamp'].min()
        self.all_connections = self.interaction_df.pivot_table(index=['user_id', 'video_id'], values=['watch_ratio', 'timestamp'])
        # set a compound network to the same shape as all but empty

    def connections_at_t(self, t):
        return self.all_connections[self.all_connections['timestamp'] == t]

    def compound_at_t(self, t):
        return self.all_connections[self.all_connections['timestamp'] <= t]

    def possible_at_t(self, t):
        # all connections that can happen and did not happen yet
        return self.all_connections[self.all_connections['timestamp'] >= t | self.all_connections['timestamp'].isna()]

    def video_user_features_t(self, t):
        compound_at_t = self.compound_at_t(t)
        merged = compound_at_t.reset_index().merge(self.user_features, on='user_id', how='left')
        video_features = merged.groupby('video_id').mean()

        return video_features


if __name__ == "__main__":
    pd.set_option('display.max_columns',  None)
    df = pd.read_csv(os.path.join('..', 'data', 'raw', 'big_matrix.csv'))
    features = pd.read_csv(os.path.join('..', 'data', 'raw', 'user_features.csv'))
    print(features[features['user_id'] == 2783])
    network = BiNetwork(df, features)
    print(network.all_connections)
    print(network.connections_at_t(0))
    print(network.compound_at_t(0))
    print(network.video_user_features_t(0))

