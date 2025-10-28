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
        # Store similar users in hashmap
        self.neighbor_dict = {}
        for user_id in interactions['user_id'].unique():
            similar_users = self.get_neighbors(user_id, threshold=0.5)
            self.neighbor_dict[user_id] = similar_users

    def get_neighbors(self, user_id: int, threshold=0.5) -> list[int]:
        """ Get neighbor users from the similarity matrix"""
        neighbors = self.similarity_df[self.similarity_df['user_id'] == user_id]
        return neighbors[neighbors['similarity'] >= threshold]["neighbor_id"].tolist()

    def get_neighbors_who_watched(self, watched_by: dict[int, set], user_id: int, video_id: int) -> set[int]:
        """ Get neighbors of user_id in similarity matrix who watched video_id for the 
        current compound network for time t."""
        neighbors = self.neighbor_dict.get(user_id, pd.Series())
        neighbors_who_watched = watched_by.get(video_id, set()).intersection(neighbors)

        return neighbors_who_watched


if __name__ == "__main__":
    small_matrix = pd.read_csv(os.path.join('.', 'data', 'raw', 'small_matrix.csv'))
    pd.set_option('display.max_columns',  None)
    network = SimNetwork(small_matrix)
    print(network.similarity_df)
