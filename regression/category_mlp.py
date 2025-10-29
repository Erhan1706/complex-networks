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


class MLPCat:

    def __init__(self, small_matrix, big_matrix, video_categories, user_features, ro=0.95, checkpoint_interval=100,
                 checkpoint_path='MLP_reg_checkpoint'):
        """
        For each time step we need a df of:
        - all possible user-video connections at t
        - feature transformed from a user features and video features pair
        I dont think preparing it for an arbitrary t, as I did in binetwok, is feasible
        therefore, start, step
        """
        # hyperparameters
        self.ro = ro
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path

        self.t = 0
        self.rmses = []
        self.positive_rmse = []
        self.negative_rmse = []
        self.rmse_timesteps = []

        # change video feature into one hot encoded features
        df = video_categories
        df['feat'] = df['feat'].apply(lambda x: x[1:-1].split(', '))
        df_expanded = df.explode('feat')
        df_onehot = pd.get_dummies(df_expanded, columns=['feat'], prefix='feature')
        df_onehot = df_onehot.groupby('video_id').max().reset_index()
        feature_cols = [col for col in df_onehot.columns if col.startswith('feature_')]
        df_onehot[feature_cols] = df_onehot[feature_cols].astype(int)
        self.video_features = df_onehot
        self.feature_columns = feature_cols

        # bi network
        self.bi_network = BiNetwork(big_matrix, user_features, filter_na_users=True)

        # init all user relevant variables
        self.all_users = set(self.bi_network.all_connections['user_id'].unique())

        # vector representing the activation for each category
        user_video_categories = pd.DataFrame(columns=['user_id'])
        user_video_categories['user_id'] = list(self.all_users)
        user_video_categories[feature_cols] = 0
        self.user_video_categories = user_video_categories.set_index('user_id')

        print(self.user_video_categories)

        # init the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', self.device)

        self.model = nn.Sequential(
            nn.Linear(62, 1),
            nn.SELU(),
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def step(self, connections_at_t):
        merged = connections_at_t.merge(self.video_features, on='video_id', how='left')

        # update user video categries with dual rate decay

        # todo maybe decay only those in connections_at_t
        # first decay all
        self.user_video_categories[self.feature_columns] *= self.ro

        # todo add increase based on watch ratio?
        # in this version we set the categories of seen connections to 1 no matter the watch ratio
        # one row per user with all categories he watched this timestep
        merged = merged.groupby('user_id')[self.feature_columns].max().reset_index()
        for _, row in merged.iterrows():
            user_id = row['user_id']
            old_features = self.user_video_categories.loc[user_id]
            new_features = (1-row[self.feature_columns]) * old_features + row[self.feature_columns]
            self.user_video_categories.loc[user_id, self.feature_columns] = new_features
        self.t += 1

    def train_step(self, train_connections):
        self.model.train(True)
        train_connections = train_connections.merge(
            self.user_video_categories.reset_index(),
            on='user_id',
            how='left',
            suffixes=('', '_user')
        )

        train_connections = train_connections.merge(
            self.video_features,
            on='video_id',
            how='left',
            suffixes=('', '_video')
        )

        # prepare input features
        X_train = train_connections[train_connections.columns[4:]].values
        y_train = train_connections['watch_ratio'].values
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(X_tensor)
        loss = nn.MSELoss()(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()
        print(f"Training loss: {loss.item()}")

    def predict(self, pred_connections):
        self.model.eval()
        pred_connections = pred_connections.merge(
            self.user_video_categories.reset_index(),
            on='user_id',
            how='left',
            suffixes=('', '_user')
        )

        pred_connections = pred_connections.merge(
            self.video_features,
            on='video_id',
            how='left',
            suffixes=('', '_video')
        )

        X_pred = pred_connections[pred_connections.columns[4:]].values
        X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).to(self.device)
        predictions = self.model(X_pred_tensor).detach().cpu().numpy()
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
    pd.set_option('display.max_columns',  None)
    try:
        small_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'small_matrix.csv'))
        big_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'big_matrix.csv'))
        video_categories = pd.read_csv(os.path.join('..', 'data', 'raw', 'item_categories.csv'))
        user_features = pd.read_csv(os.path.join('..', 'data', 'raw', 'user_features.csv'))
    except:
        small_matrix = pd.read_csv(os.path.join('data', 'raw', 'small_matrix.csv'))
        big_matrix = pd.read_csv(os.path.join('data', 'raw', 'big_matrix.csv'))
        video_categories = pd.read_csv(os.path.join('data', 'raw', 'item_categories.csv'))
        user_features = pd.read_csv(os.path.join('data', 'raw', 'user_features.csv'))
    print('data read in')

    mlp = MLPCat(small_matrix, big_matrix, video_categories, user_features,
                   checkpoint_interval=500, checkpoint_path='mlp_reg_checkpoint')

    # rmses = model.train(start=1000, stop=7000)

    #with open('mlp_rmses.pkl', 'wb') as f:
     #   pickle.dump(rmses, f)