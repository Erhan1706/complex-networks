"""Quick test to find optimal similarity threshold"""
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

print("Loading data...")
interactions = pd.read_csv('data/raw/small_matrix.csv')
user_video_matrix = interactions.pivot_table(
    index='user_id',
    columns='video_id',
    values='watch_ratio',
    fill_value=0
)

print("Computing similarity matrix...")
similarity_matrix = cosine_similarity(user_video_matrix.values)
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=user_video_matrix.index,
    columns=user_video_matrix.index
)

users = user_video_matrix.index.tolist()

print("\nTesting different thresholds:\n")
print(f"{'Threshold':<12} {'Edges':<10} {'Density':<10} {'Isolated':<12} {'Giant %':<10} {'Status':<15}")
print("-" * 75)

for threshold in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
    G = nx.Graph()
    G.add_nodes_from(users)

    # Add edges above threshold
    for i, user_i in enumerate(users):
        for j, user_j in enumerate(users[i+1:], start=i+1):
            sim = similarity_df.loc[user_i, user_j]
            if sim >= threshold:
                G.add_edge(user_i, user_j, weight=sim)

    density = nx.density(G)
    isolated = sum(1 for node in G.nodes() if G.degree(node) == 0)
    isolated_pct = isolated / G.number_of_nodes() * 100

    # Find giant component
    if G.number_of_edges() > 0:
        components = list(nx.connected_components(G))
        giant = max(components, key=len)
        giant_pct = len(giant) / G.number_of_nodes() * 100
    else:
        giant_pct = 0

    # Determine status
    if isolated_pct > 20:
        status = "Too fragmented"
    elif density > 0.6:
        status = "Too dense"
    elif giant_pct < 70:
        status = "Too sparse"
    else:
        status = "✓ GOOD"

    print(f"{threshold:<12.2f} {G.number_of_edges():<10,} {density:<10.4f} {isolated:<5} ({isolated_pct:4.1f}%) {giant_pct:<10.1f} {status:<15}")

print("\nRecommendation: Choose threshold where:")
print("  - Isolated nodes < 10%")
print("  - Giant component > 80%")
print("  - Density 0.2-0.5 (good balance)")
