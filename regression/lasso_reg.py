import os
import signal
import sys
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDRegressor
from sim_network import SimNetwork
from bi_network import BiNetwork
from graphing import *
from collections import defaultdict

class LassoReg:

    def __init__(self, small_matrix, big_matrix, user_features, ro=0.95, checkpoint_interval=100,
                 checkpoint_path='lasso_reg_checkpoint'):
        """
        For each time step we need a df of:
        - all possible user-video connections at t
        - feature transformed from a user features and video features pair
        I dont think preparing it for an arbitrary t, as I did in binetwok, is feasible
        therefore, start, step
        """
        self.bi_network = BiNetwork(big_matrix, user_features)
        print('BiNetwork created.')
        self.sim_network = SimNetwork(small_matrix)
        print('SimNetwork created.')
        self.user_features = user_features
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path
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
        self.rmses = []
        self.rmse_timesteps = []
        self.watched_by = defaultdict(set)

        self.start_features()

    def start_features(self):
        # get all possible connections at t=0, full mesh
        print('Initializing LassoReg model.')
        possible = self.bi_network.all_connections
        self.user_feature_columns = [col for col in self.user_features.columns if col[:6] == 'onehot']

        # todo maybe standardizing here is not good
        # standardize user features
        # change user_id as index back and forth to not normalize on it
        self.user_features = self.user_features.set_index('user_id', drop=True)
        self.user_features = self.user_features[self.user_feature_columns]

        self.user_features[self.user_feature_columns] = (self.user_features[self.user_feature_columns] - self.user_features[
            self.user_feature_columns].mean()) / self.user_features[self.user_feature_columns].std()

        self.user_features = self.user_features.reset_index(names='user_id')

        # 0 is the pop mean for standardized features
        videos = possible['video_id'].unique()
        self.video_features = pd.DataFrame(0.0, index=videos, columns=self.user_feature_columns)
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

    def add_similarity_feature(self, df, cap=50):
        similar_counts = []
        for _, row in df.iterrows():
            user_id, video_id = row['user_id'], row['video_id']
            neighbors_watched = self.sim_network.get_neighbors_who_watched(
                self.watched_by, user_id, video_id
            )
            similar_counts.append(len(neighbors_watched))

        # standardize
        similar_counts = np.array(similar_counts)
        m = similar_counts.mean()
        s = similar_counts.std()
        if s < 1e-8:
            standardized = similar_counts - m  # if all zero, just zero-mean
        else:
            standardized = (similar_counts - m) / s

        df['similar_users_watched'] = standardized
        return df

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
        self.add_similarity_feature(train_connections)
        self.feature_columns = [col for col in train_connections.columns if col[:6] == 'onehot']
        self.feature_columns.append('similar_users_watched')
        X_train = train_connections[self.feature_columns].fillna(0).values
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
        self.add_similarity_feature(pred_connections)
        X_pred = pred_connections[self.feature_columns].fillna(0).values
        print(f"Predicting at t={self.t} on {pred_connections.shape[0]} samples with {len(self.feature_columns)} features.")
        print(X_pred)

        predictions = self.reg_model.predict(X_pred)
        true = pred_connections['watch_ratio'].values

        return predictions, true

    def eval(self, predictions, true):
        # todo more advanced metric that takes the not connected into account?
        # simple RMSE

        mse = np.mean((predictions - true) ** 2)
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse}")

        return rmse

    def checkpoint(self):
        with open(f'{self.checkpoint_path}_t{self.t}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def train(self, start=1000, stop=7000):

        # no predict for t = 0
        self.t = start
        connections = self.bi_network.connections_at_t(self.t)
        if connections.size != 0:
            self.step(connections)
            self.train_step(connections)
        else:
            self.t += 1
        self.rmses = []

        # Build historical watched_by from data before start
        compound = self.bi_network.compound_at_t(start)
        for video_id in compound['video_id'].unique():
            users_who_watched = compound[compound['video_id'] == video_id]['user_id']
            self.watched_by[video_id].update(users_who_watched)

        while self.t < stop:
            print(f"Time step {self.t}")
            connections = self.bi_network.connections_at_t(self.t)
            predictions, true = self.predict(connections)
            self.rmses.append(self.eval(predictions, true))
            self.rmse_timesteps.append(self.t)

            for video_id in connections['video_id'].unique():
                # Keep track of what videos were watched at the current t
                users_who_watched = connections[connections['video_id'] == video_id]['user_id']
                self.watched_by[video_id].update(users_who_watched)

            if connections.size != 0:
                self.step(connections)
                self.train_step(connections)
            else:
                self.t += 1

            if self.t % self.checkpoint_interval == 0:
                self.checkpoint()

        print('Training complete., final checkpointing...')
        self.checkpoint()

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

        lasso_reg = LassoReg(small_matrix, big_matrix, features, checkpoint_interval=1000)
        rmses = lasso_reg.train()
        plot_rmse([rmses], ['Lasso Regression'], title='Lasso Regression RMSE over Time')
        print('finished')
    except Exception as e:
        print(f'An error occurred: {e}')
        if lasso_reg is not None:
            plot_rmse([lasso_reg.rmses], ['Lasso Regression'], title='Lasso Regression RMSE over Time')
        sys.exit(1)
