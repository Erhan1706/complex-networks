import os

import numpy as np
import pandas as pd

from sim_network import SimNetwork
from bi_network import BiNetwork
from sklearn.linear_model import SGDRegressor


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
        self.reg_model = SGDRegressor(
            penalty='l1',
            alpha=0.001,
            max_iter=1000,
            tol=1e-3,
            warm_start=True
        )

        self.possible = None
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
        self.possible = possible

        self.train()

    def step(self, connections_at_t):
        # update video features based on connections at t
        # using exponentially weighted moving average with bias correction
        if connections_at_t.empty:
            self.t += 1
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
        # set possible connections
        self.possible = self.bi_network.possible_at_t(self.t)

    def train_step(self, train_connections):

        if train_connections.empty:
            return

        train_connections = train_connections.merge(
            self.user_features,
            on='user_id',
            how='left',
            suffixes=('', '_user')
        )

        train_connections = train_connections.merge(
            self.video_features.reset_index(drop=False, names='video_id'),
            on='video_id',
            how='left',
            suffixes=('', '_video')
        )

        feature_columns = [col for col in train_connections.columns if col[:6] == 'onehot']
        print(feature_columns)
        X_train = train_connections[feature_columns].values
        y_train = train_connections['watch_ratio'].values
        print(f"Training at t={self.t-1} on {X_train.shape[0]} samples with {X_train.shape[1]} features.")
        print(X_train)
        print(y_train)
        self.reg_model.partial_fit(X_train, y_train)

    def predict(self, pred_connections):

        #todo is it reasonable to predict for all possible connections at t?

        if pred_connections.empty:
            return np.array([]), np.array([])
        pred_connections = pred_connections.merge(
            self.user_features,
            on='user_id',
            how='left',
            suffixes=('', '_user')
        )

        pred_connections = pred_connections.merge(
            self.video_features.reset_index(drop=False, names='video_id'),
            on='video_id',
            how='left',
            suffixes=('', '_video')
        )
        feature_columns = [col for col in pred_connections if col[:6] == 'onehot']
        X_pred = self.possible[feature_columns].values
        print(f"Predicting at t={self.t} on {self.possible.shape[0]} samples with {len(feature_columns)} features.")
        print(X_pred)

        predictions = self.reg_model.predict(X_pred)
        true = self.possible['watch_ratio'].values

        return predictions, true

    def eval(self, predictions, true):
        # todo more advanced metric that takes the not connected into account?
        # simple RMSE

        mask = ~np.isnan(true)
        mse = np.mean((predictions[mask] - true[mask]) ** 2)
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse}")

        return rmse

    def train(self):

        # no predict for t = 0
        # todo this is messy, it would be nice to clean the empty time steps
        connections = self.bi_network.connections_at_t(self.t)
        # todo 10 for testing to make sure it works, change later to max(timestamp)
        for i in range(10):
            self.step(connections)
            self.train_step(connections)
            # t incrementd in step
            connections = self.bi_network.connections_at_t(self.t)
            predictions, true = self.predict(connections)
            if predictions.size == 0:
                continue
            self.eval(predictions, true)


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    small_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'small_matrix.csv'))
    big_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'big_matrix.csv'))
    features = pd.read_csv(os.path.join('..', 'data', 'raw', 'user_features.csv'))

    lasso_reg = LassoReg(small_matrix, big_matrix, features)
    a=1
    print('finished')
