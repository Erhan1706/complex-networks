import os
import signal
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDRegressor
from sim_network import SimNetwork
from bi_network import BiNetwork
from graphing import *


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
        print('BiNetwork created.')
        # self.sim_network = SimNetwork(small_matrix)
        print('SimNetwork created.')
        self.user_features = user_features
        self.ro = ro
        self.reg_model = SGDRegressor(
            penalty='l1',
            alpha=0.001,
            max_iter=100,
            tol=1e-3,
            warm_start=True
        )

        self.possible = None
        self.video_features = None

        self.t = 0

        self.start_features()

    def start_features(self):
        # get all possible connections at t=0, full mesh
        print('Initializing LassoReg model.')
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
        # initialize all videos with population mean (not 0)
        """
        self.population_video_features = self.user_features.drop('user_id', axis=1).mean() 
        self.video_features = pd.DataFrame(
            [self.population_video_features.values] * len(videos),
            index=videos,
            columns=user_feature_columns
        ) """
        self.possible = possible

        print('Initialized LassoReg model.')

    def step(self, connections_at_t):
        # update video features based on connections at t
        # using exponentially weighted moving average with bias correction
        if connections_at_t.empty:
            self.t += 1
            # still update "possible" connections after time moves forward
            self.possible = self.bi_network.possible_at_t(self.t)
            return

        # user features 
        merged = connections_at_t.reset_index(drop=True).merge(
            self.user_features, on='user_id', how='left'
        )

        #todo maybe weight them by watch ratio in the future?
        feature_cols = [c for c in merged.columns if c.startswith("onehot")]
        video_user_features = merged.groupby("video_id")[feature_cols].mean()

        # update with no global denominator
        for vid, new_vec in video_user_features.iterrows():
            old_vec = self.video_features.loc[vid]
            updated_vec = self.ro * old_vec + (1 - self.ro) * new_vec
            self.video_features.loc[vid] = updated_vec

        self.t += 1
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
        #print(feature_columns)
        X_train = train_connections[feature_columns].fillna(0).values
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
        X_pred = pred_connections[feature_columns].fillna(0).values
        print(f"Predicting at t={self.t} on {pred_connections.shape[0]} samples with {len(feature_columns)} features.")
        print(X_pred)

        predictions = self.reg_model.predict(X_pred)
        true = pred_connections['watch_ratio'].values

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
        connections = self.bi_network.connections_at_t(self.t)
        self.rmses = []
        for i in range(2000, int(self.bi_network.all_connections['timestamp'].max() + 1)): 
            print(i)
            print(connections)
            if connections.size != 0:
                self.step(connections)
                self.train_step(connections)
            else:
                self.t += 1
            # t incrementd in step
            connections = self.bi_network.connections_at_t(self.t)
            predictions, true = self.predict(connections)
            if predictions.size == 0:
                continue
            self.rmses.append(self.eval(predictions, true))
        return self.rmses


if __name__ == "__main__":
    lasso_reg = None
    
    # Save plot on interrupt
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully"""
        print('\n\nInterrupt received! Saving results...')
        if lasso_reg is not None:
            plot_rmse([lasso_reg.rmses], ['Lasso Regression'], title='Lasso Regression RMSE over Time')
        print('Results saved. Exiting.')
        sys.exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    try:
        pd.set_option('display.max_columns', None)
        small_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'small_matrix.csv'))
        print('small matrix read in')
        big_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'big_matrix.csv'))
        print('big matrix read in')
        features = pd.read_csv(os.path.join('..', 'data', 'raw', 'user_features.csv'))
        print('data read in')

        lasso_reg = LassoReg(small_matrix, big_matrix, features)
        rmses = lasso_reg.train()
        plot_rmse([rmses], ['Lasso Regression'], title='Lasso Regression RMSE over Time')
        print('finished')
    except Exception as e:
        print(f'An error occurred: {e}')
        if lasso_reg is not None:
            plot_rmse([lasso_reg.rmses], ['Lasso Regression'], title='Lasso Regression RMSE over Time')
        sys.exit(1)
