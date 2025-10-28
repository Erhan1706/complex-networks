import os
import signal
import sys
import pickle
import itertools

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.neural_network import MLPRegressor
from sim_network import SimNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from bi_network import BiNetwork
from graphing import *


class MLPReg:

    def __init__(self, small_matrix, big_matrix, user_features, ro=0.95, checkpoint_interval=100,
                 checkpoint_path='MLP_reg_checkpoint'):
        """
        For each time step we need a df of:
        - all possible user-video connections at t
        - feature transformed from a user features and video features pair
        I dont think preparing it for an arbitrary t, as I did in binetwok, is feasible
        therefore, start, step
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', self.device)
        self.bi_network = BiNetwork(big_matrix, user_features, filter_na_users=True)
        print('BiNetwork created.')
        # self.sim_network = SimNetwork(small_matrix)
        print('SimNetwork created.')
        self.user_features = user_features
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path
        self.ro = ro

        self.model = self.net = nn.Sequential(
            nn.Linear(36, 1),
            nn.SELU(),
        ).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        self.video_features = None

        self.t = 0
        self.rmses = []
        self.positive_rmse = []
        self.negative_rmse = []
        self.rmse_timesteps = []

        self.all_users = set(self.bi_network.user_features['user_id'].unique())
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

        print('Initialized LassoReg model.')

    def step(self, connections_at_t):
        # update video features based on connections at t
        # using exponentially weighted moving average with bias correction
        if connections_at_t.empty:
            self.t += 1
            # still update "possible" connections after time moves forward
            return

        # user features
        merged = connections_at_t.reset_index(drop=True).merge(
            self.user_features, on='user_id', how='left'
        )

        # todo maybe weight them by watch ratio in the future?
        feature_cols = [c for c in merged.columns if c.startswith("onehot")]
        video_user_features = merged.groupby("video_id")[feature_cols].mean()

        # update with no global denominator
        for vid, new_vec in video_user_features.iterrows():
            old_vec = self.video_features.loc[vid]
            updated_vec = self.ro * old_vec + (1 - self.ro) * new_vec
            self.video_features.loc[vid] = updated_vec

        self.t += 1

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
        # print(feature_columns)
        X_train = train_connections[feature_columns].fillna(0).values
        y_train = train_connections['watch_ratio'].values
        print(f"Training at t={self.t - 1} on {X_train.shape[0]} samples with {X_train.shape[1]} features.")
        self.model.train()
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(X_tensor)
        loss = nn.MSELoss()(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()
        print(f"Training loss: {loss.item()}")

    def predict(self, pred_connections):

        # todo is it reasonable to predict for all possible connections at t?

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

        X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).to(self.device)
        predictions = self.model(X_pred_tensor).detach().cpu().numpy()
        print(predictions)
        true = pred_connections['watch_ratio'].values

        return predictions, true

    def eval(self, predictions, true):
        # todo more advanced metric that takes the not connected into account?
        # simple RMSE

        positive_true = true[true >= 0]
        positive_predictions = predictions[true >= 0]
        negative_true = true[true < 0]
        negative_predictions = predictions[true < 0]
        positive_rmse = np.sqrt(np.mean((positive_predictions - positive_true) ** 2))
        negative_rmse = np.sqrt(np.mean((negative_predictions - negative_true) ** 2))
        mse = np.mean((predictions - true) ** 2)
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse}")

        return positive_rmse, negative_rmse, rmse

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

        while self.t < stop:
            print(f"Time step {self.t}")
            connections = self.bi_network.connections_at_t(self.t)

            # create negative_samples
            unique_videos = connections['video_id'].unique()
            unique_users = connections['user_id'].unique()
            valid_users = self.all_users - set(unique_users)
            pairs = list(itertools.product(valid_users, unique_videos))
            n_negative = len(connections)
            print(f'Generating {n_negative} negative samples from {len(pairs)} possible pairs.')
            random_subset = np.random.choice(len(pairs), size=n_negative, replace=False)
            negative_samples = pd.DataFrame([pairs[i] for i in random_subset], columns=['user_id', 'video_id'])
            negative_samples['watch_ratio'] = -1.7581

            # add negative to all
            connections = pd.concat([connections, negative_samples], ignore_index=True)

            predictions, true = self.predict(connections)
            positive_rmse, negative_rmse, rmse = self.eval(predictions, true)
            self.rmses.append(rmse)
            self.positive_rmse.append(positive_rmse)
            self.negative_rmse.append(negative_rmse)

            self.rmse_timesteps.append(self.t)
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
    pd.set_option('display.max_columns', None)
    small_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'small_matrix.csv'))
    print('small matrix read in')
    big_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'big_matrix.csv'))
    print('big matrix read in')
    features = pd.read_csv(os.path.join('..', 'data', 'raw', 'user_features.csv'))
    print('data read in')

    mlp = MLPReg(small_matrix, big_matrix, features, checkpoint_interval=10)
    rmses = mlp.train()
    plot_rmse([rmses], ['MLP iwth negative'], title='MLP RMSE over Time')
    print('finished')
