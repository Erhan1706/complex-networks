import os

import numpy as np
import pandas as pd

from sim_network import SimNetwork
from bi_network import BiNetwork
from sklearn.linear_model import Lasso


class LassoReg:

    def __init__(self, small_matrix, big_matrix, user_features, ro=0.95):
        """
        For each time step we need a df of:
        - all possible user-video connections at t
        - feature transformed from a user features and video features pair
        I dont think preparing it for an arbitrary t, as I did in binetwok, is feasible
        therefore, start, step
        """
        self.bi_network = BiNetwork(big_matrix, user_features)
        self.sim_network = SimNetwork(small_matrix)
        self.user_features = user_features
        self.ro = ro
        self.reg_model = Lasso(alpha=0.1)

        self.connections_df = None
        self.video_features = None

        self.t = 0

        self.start_features()

    def start_features(self):
        # get all possible connections at t=0, full mesh
        possible = self.bi_network.all_connections
        user_feature_columns = [col for col in self.user_features.columns if col[:6] == 'onehot']

        # todo maybe standardizing here is not good
        # standardize user features
        # change user_id as index back and forth to not normalize on it
        self.user_features = self.user_features.set_index('user_id', drop=True)
        self.user_features = self.user_features[user_feature_columns]
        self.user_features[user_feature_columns] = (self.user_features[user_feature_columns] - self.user_features[
            user_feature_columns].mean()) / self.user_features[user_feature_columns].std()
        self.user_features = self.user_features.reset_index(names='user_id')

        # 0 is the pop mean for standardized features
        videos = possible['video_id'].unique()
        self.video_features = pd.DataFrame(0.0, index=videos, columns=user_feature_columns)
        self.connections_df = possible

    def step(self):
        # update video features based on connections at t
        # using exponentially weighted moving average with bias correction
        connections_at_t = self.bi_network.connections_at_t(self.t)
        if connections_at_t.empty:
            return
        # the values to update with is the mean of user features of users who watched the video at t
        merged = connections_at_t.reset_index(drop=True).merge(self.user_features, on='user_id', how='left')

        #todo maybe weight them by watch ratio in the future?
        merged = merged.drop(columns=['user_id', 'timestamp', 'watch_ratio'])
        video_user_features = merged.groupby('video_id').mean()

        # update
        changing_videos = self.video_features.loc[video_user_features.index]
        self.video_features.loc[video_user_features.index] = \
            (changing_videos * self.ro + video_user_features * (1 - self.ro))/(1 - self.ro ** (self.t + 1))

        self.t += 1

        # set possible concetions
        self.connections_df = self.bi_network.possible_at_t(self.t)


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    small_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'small_matrix.csv'))
    big_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'big_matrix.csv'))
    features = pd.read_csv(os.path.join('..', 'data', 'raw', 'user_features.csv'))

    lasso_reg = LassoReg(small_matrix, big_matrix, features)
    a=1
    print('finished')
