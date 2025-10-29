from regression.mlp_feat_only import MLPReg
import os
import itertools
import numpy as np
import pandas as pd
from graphing import *
import signal
import sys
import pickle

class Baseline0(MLPReg):
    def __init__(self, big_matrix, user_features, checkpoint_interval=50, checkpoint_path='baseline_0_checkpoint'):
        super().__init__(None, big_matrix, user_features, checkpoint_interval=checkpoint_interval,
                         checkpoint_path=checkpoint_path)

    def step(self, connections_at_t):
        # step_timestep
        self.t += 1
        # Baseline 0 does not update any features

    def train(self, start=1000, stop=7000):
        self.t = start
        while self.t < stop:
            self.step(None)
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
            predictions = pd.Series([0 for _ in range(len(connections))])
            true = connections['watch_ratio']
            positive_rmse, negative_rmse, rmse = self.eval(predictions, true)

            self.rmses.append(rmse)
            self.positive_rmse.append(positive_rmse)
            self.negative_rmse.append(negative_rmse)

            if self.t % self.checkpoint_interval == 0:
                self.checkpoint()

        return self.rmses


if __name__ == "__main__":
    baseline = None

    # Save plot on interrupt
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully"""
        print('\n\nInterrupt received! Saving results...')
        if baseline is not None:
            plot_rmse([baseline.rmses], ['Lasso Regression'], title='Lasso Regression RMSE over Time')
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

    baseline = Baseline0(big_matrix, features, checkpoint_interval=1000)
    rmses = baseline.train()
    plot_rmse([baseline.positive_rmse, baseline.negative_rmse, baseline.rmses],
              ['Positive RMSE', 'Negative RMSE', 'Overall RMSE'],
              title='Baseline 0 RMSE over Time')
