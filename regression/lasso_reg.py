import os

import numpy as np
import pandas as pd

from sim_network import SimNetwork
from bi_network import BiNetwork
from sklearn.linear_model import Lasso


class LassoReg:

    def __init__(self, small_matrix, big_matrix, user_features):
        self.bi_network = BiNetwork(big_matrix, user_features)
        self.sim_network = SimNetwork(small_matrix)
        self.user_features = user_features
        self.create_features(t=100)
        self.reg_model = Lasso(alpha=0.1)

    def create_features(self, t):
        possible = self.bi_network.possible_at_t(t)
        print(possible.head())
        video_features = self.bi_network.video_user_features_t(t)
        all_options = possible.merge(video_features, on='video_id', how='left', suffixes=('', '_video'))
        all_options = all_options.merge(self.user_features, on='user_id', how='left', suffixes=('', '_user'))
        print(all_options.head())
        vide_feature_columns = [col for col in all_options.columns if col[-5:] != '_user' and col[:6] == 'onehot']
        new_feature_columns = [col+'_new' for col in vide_feature_columns]
        user_feature_columns = [col for col in all_options.columns if col[-5:] == '_user' and col[:6] == 'onehot']
        print(new_feature_columns)
        print(vide_feature_columns)
        print(user_feature_columns)
        print(len(vide_feature_columns), len(user_feature_columns))
        all_options[new_feature_columns] = all_options[vide_feature_columns].values


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    small_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'small_matrix.csv'))
    big_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'big_matrix.csv'))
    features = pd.read_csv(os.path.join('..', 'data', 'raw', 'user_features.csv'))

    lasso_reg = LassoReg(small_matrix, big_matrix, features)
    lasso_reg.create_features(t=10)
