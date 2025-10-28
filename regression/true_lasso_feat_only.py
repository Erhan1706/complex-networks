import sys
import os
import pandas as pd

from regression.lasso_reg import LassoReg
from regression.bi_network import BiNetwork
from graphing import *
from sklearn.linear_model import SGDRegressor, Lasso
import signal
import sys
import pickle


class TrueLasso(LassoReg):
    def __init__(self, small_matrix, big_matrix, user_features, checkpoint_interval=50,
                 checkpoint_path='true_lasso_checkpoint', ro=0.9):
        super().__init__(small_matrix, big_matrix, user_features, ro=ro, checkpoint_interval=checkpoint_interval,
                         checkpoint_path=checkpoint_path)

        self.reg_model = Lasso(
            alpha=0.01,
            max_iter=1000,
            warm_start=True
        )
        self.historical_connections = pd.DataFrame(columns=['user_id', 'video_id', 'timestamp', 'watch_ratio'])

    def step(self, connections_at_t):
        # step_timestep
        self.t += 1

        # update seen connections
        self.historical_connections = pd.concat([self.historical_connections, connections_at_t], ignore_index=True)

        # get video features at t
        merged = connections_at_t.reset_index(drop=True).merge(
            self.user_features, on='user_id', how='left'
        )

        feature_cols = [c for c in merged.columns if c.startswith("onehot")]
        video_user_features = merged.groupby("video_id")[feature_cols].mean()

        # update video features with exponential moving average
        for vid, new_vec in video_user_features.iterrows():
            old_vec = self.video_features.loc[vid]
            updated_vec = self.ro * old_vec + (1 - self.ro) * new_vec
            self.video_features.loc[vid] = updated_vec

    def train_step(self, train_connections):

        # for true lasso, train on all historical connections up to time instead
        # of train connections only

        train_connections = self.historical_connections.merge(
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

        feature_cols = [c for c in train_connections.columns if c.startswith("onehot")]

        print(f"Training Lasso on {len(train_connections)} connections at time {self.t}")

        X_train = train_connections[feature_cols].fillna(0).values
        y_train = train_connections['watch_ratio'].values

        self.reg_model.fit(X_train, y_train)


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
    pd.set_option('display.max_columns', None)
    small_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'small_matrix.csv'))
    print('small matrix read in')
    big_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'big_matrix.csv'))
    print('big matrix read in')
    features = pd.read_csv(os.path.join('..', 'data', 'raw', 'user_features.csv'))
    print('data read in')

    lasso_reg = TrueLasso(small_matrix, big_matrix, features, checkpoint_interval=500,
                          checkpoint_path='true_lasso/true_lasso_checkpoint')
    rmses = lasso_reg.train(stop=5000)
    plot_rmse([rmses], ['Lasso Regression'], title='Lasso Regression RMSE over Time')
    print('finished')