import pandas as pd
import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity


class SimNetwork:

    def __init__(self, interactions: pd.DataFrame):
        user_video_matrix = interactions.pivot_table(
            index='user_id',
            columns='video_id',
            values='watch_ratio',
            fill_value=0  # If user didn't watch, assume 0 (rare with 99.7% coverage)
        )
        similarity_matrix = cosine_similarity(user_video_matrix.values)
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=user_video_matrix.index,
            columns=user_video_matrix.index
        )
        self.similarity_df = similarity_df.reset_index().melt(id_vars='user_id', var_name='neighbor_id', value_name='similarity')
        self.similarity_df = self.similarity_df[self.similarity_df['user_id'] != self.similarity_df['neighbor_id']].reset_index(drop=True)

    def get_neighbors(self, user_ids: pd.Series, threshold=0.5):
        candidate_users = self.similarity_df[self.similarity_df['user_id'].isin(user_ids)]
        return candidate_users[candidate_users['similarity'] >= threshold]


if __name__ == "__main__":
    small_matrix = pd.read_csv(os.path.join('..', 'data', 'raw', 'small_matrix.csv'))
    pd.set_option('display.max_columns',  None)
    network = SimNetwork(small_matrix)
    print(network.similarity_df)
    print(network.get_neighbors(pd.Series([19, 7141]), threshold=0.1))