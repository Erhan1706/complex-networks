"""
User Similarity Network Construction and Analysis

This script builds a user similarity network from viewing behavior and analyzes
its properties using standard complex network metrics.

Network Definition:
- Nodes: Users (N = 1,411)
- Edges: Connect users with similar viewing patterns
- Edge criterion: Cosine similarity of watch_ratio vectors > threshold
- Network type: Undirected, weighted (similarity as weight)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import os

# Plotting setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Configuration
SIMILARITY_THRESHOLD = 0.7  # Adjust this to control edge density

# Construct path relative to this script's location (works regardless of where you run it from)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'big_matrix.csv')

print("="*70)
print("USER SIMILARITY NETWORK ANALYSIS")
print("="*70)
print()

# ============================================================================
# 1. LOAD DATA AND CREATE USER-VIDEO MATRIX
# ============================================================================
print("STEP 1: Loading data and creating user-video matrix")
print("-"*70)

# Load interaction data
print("Loading interaction data...")
interactions = pd.read_csv(DATA_PATH)

print(f"Total interactions: {len(interactions):,}")
print(f"Unique users: {interactions['user_id'].nunique():,}")
print(f"Unique videos: {interactions['video_id'].nunique():,}")

# Create user-video matrix
# Rows = users, Columns = videos, Values = watch_ratio
print("\nCreating user-video matrix...")
user_video_matrix = interactions.pivot_table(
    index='user_id',
    columns='video_id',
    values='watch_ratio',
    fill_value=0  # If user didn't watch, assume 0 (rare with 99.7% coverage)
)

print(f"Matrix shape: {user_video_matrix.shape[0]:,} users × {user_video_matrix.shape[1]:,} videos")
print()

# ============================================================================
# 2. COMPUTE USER SIMILARITY
# ============================================================================
print("STEP 2: Computing pairwise user similarity")
print("-"*70)
print("Computing cosine similarity between all users...")
print("(This may take a minute for 1,411 users)")
print()

# Compute pairwise cosine similarity
# Cosine similarity: similarity(u_i, u_j) = dot(v_i, v_j) / (||v_i|| * ||v_j||)
similarity_matrix = cosine_similarity(user_video_matrix.values)

print(f"Similarity matrix shape: {similarity_matrix.shape}")
print(f"  ({similarity_matrix.shape[0]:,} users × {similarity_matrix.shape[1]:,} users)")

# Convert to DataFrame for easier handling
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=user_video_matrix.index,
    columns=user_video_matrix.index
)

# Set diagonal to NaN (self-similarity not interesting)
np.fill_diagonal(similarity_matrix, np.nan)

# Analyze similarity distribution
similarities = similarity_matrix[~np.isnan(similarity_matrix)]

print(f"\nSimilarity statistics (all {len(similarities):,} user pairs):")
print(f"  Mean:   {np.mean(similarities):.4f}")
print(f"  Median: {np.median(similarities):.4f}")
print(f"  Std:    {np.std(similarities):.4f}")
print(f"  Min:    {np.min(similarities):.4f}")
print(f"  Max:    {np.max(similarities):.4f}")

# Show edge counts for different thresholds
print(f"\nEdge counts for different similarity thresholds:")
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    num_edges = np.sum(similarities > threshold)
    density = num_edges / len(similarities)
    print(f"  Threshold {threshold:.1f}: {num_edges:,} edges ({100*density:.2f}% density)")

print()

# Plot similarity distribution
plt.figure(figsize=(10, 5))
plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of User Similarity')
plt.axvline(np.mean(similarities), color='red', linestyle='--',
            linewidth=2, label=f'Mean = {np.mean(similarities):.3f}')
plt.axvline(SIMILARITY_THRESHOLD, color='green', linestyle='--',
            linewidth=2, label=f'Threshold = {SIMILARITY_THRESHOLD}')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('similarity_distribution.png', dpi=150)
print("Saved: similarity_distribution.png")
plt.close()
print()

# ============================================================================
# 3. BUILD NETWORK WITH THRESHOLD
# ============================================================================
print("STEP 3: Building network with threshold")
print("-"*70)
print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
print()

# Create graph
G = nx.Graph()

# Add all users as nodes
users = user_video_matrix.index.tolist()
G.add_nodes_from(users)
print(f"Added {G.number_of_nodes():,} nodes")

# Add edges for user pairs with similarity > threshold
print("Adding edges (this may take a moment)...")
edges_added = 0
for i, user_i in enumerate(users):
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(users)} users processed...")
    for j, user_j in enumerate(users[i+1:], start=i+1):
        sim = similarity_df.loc[user_i, user_j]
        if sim >= SIMILARITY_THRESHOLD:
            G.add_edge(user_i, user_j, weight=sim)
            edges_added += 1

print(f"Added {edges_added:,} edges (similarity ≥ {SIMILARITY_THRESHOLD})")

# Network statistics
N = G.number_of_nodes()
E = G.number_of_edges()
max_edges = N * (N - 1) / 2
density = nx.density(G)

print(f"\nNetwork Properties:")
print(f"  Nodes (N):          {N:,}")
print(f"  Edges (E):          {E:,}")
print(f"  Density:            {density:.4f} ({100*density:.2f}%)")
print(f"  Max possible edges: {int(max_edges):,}")
print(f"  Is connected:       {nx.is_connected(G)}")

if not nx.is_connected(G):
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    print(f"  Components:         {len(components)}")
    print(f"  Largest component:  {len(largest_cc)} nodes ({100*len(largest_cc)/N:.1f}%)")
    print(f"\n  Note: Using largest connected component for path-based metrics")
    G_main = G.subgraph(largest_cc).copy()
else:
    G_main = G

print()

# ============================================================================
# 4. DEGREE DISTRIBUTION P(k)
# ============================================================================
print("STEP 4: Analyzing degree distribution")
print("-"*70)

# Compute degrees
degrees = dict(G.degree())
degree_values = list(degrees.values())

print("Degree Statistics:")
print(f"  Mean degree ⟨k⟩:  {np.mean(degree_values):.2f}")
print(f"  Median degree:    {np.median(degree_values):.0f}")
print(f"  Max degree:       {np.max(degree_values)}")
print(f"  Min degree:       {np.min(degree_values)}")
print(f"  Std deviation:    {np.std(degree_values):.2f}")

# Degree distribution plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Linear scale
axes[0].hist(degree_values, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Degree k')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Degree Distribution P(k)')
axes[0].axvline(np.mean(degree_values), color='red', linestyle='--',
                linewidth=2, label=f'Mean ⟨k⟩ = {np.mean(degree_values):.1f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Log-Y scale
axes[1].hist(degree_values, bins=50, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Degree k')
axes[1].set_ylabel('Frequency (log scale)')
axes[1].set_title('Degree Distribution P(k) - Log-Y')
axes[1].set_yscale('log')
axes[1].grid(alpha=0.3)

# Log-Log scale (check for power law)
degree_counts = np.bincount(degree_values)
degrees_unique = np.arange(len(degree_counts))
mask = degree_counts > 0

axes[2].scatter(degrees_unique[mask], degree_counts[mask], alpha=0.6, s=50)
axes[2].set_xlabel('Degree k (log scale)')
axes[2].set_ylabel('P(k) (log scale)')
axes[2].set_title('Degree Distribution - Log-Log\n(Linear trend = Power Law)')
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('degree_distribution.png', dpi=150)
print("\nSaved: degree_distribution.png")
plt.close()

print("\nInterpretation:")
print("  - If log-log plot is roughly linear: Power-law (scale-free network)")
print("  - If bell-shaped on linear scale: Random-like (Poisson)")
print("  - If heavy right tail: Few hubs, many low-degree nodes")
print()

# ============================================================================
# 5. CLUSTERING COEFFICIENT C
# ============================================================================
print("STEP 5: Computing clustering coefficient")
print("-"*70)
print("Computing clustering coefficient...")

C = nx.average_clustering(G)
C_random = np.mean(degree_values) / G.number_of_nodes()  # Expected for random

print(f"\nClustering Coefficient:")
print(f"  Average clustering C:     {C:.4f}")
print(f"  Expected for random:      {C_random:.4f}")
print(f"  Ratio C / C_random:       {C / C_random:.2f}x")

if C > 2 * C_random:
    print(f"\n  -> Much higher than random -> Network has community structure")
elif C > C_random:
    print(f"\n  -> Higher than random -> Some clustering present")
else:
    print(f"\n  -> Similar to random network")

# Local clustering distribution
local_clustering = nx.clustering(G)
local_clustering_values = list(local_clustering.values())

plt.figure(figsize=(10, 5))
plt.hist(local_clustering_values, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Local Clustering Coefficient')
plt.ylabel('Frequency')
plt.title(f'Distribution of Local Clustering (⟨C⟩ = {C:.3f})')
plt.axvline(C, color='red', linestyle='--', linewidth=2, label=f'Average = {C:.3f}')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('clustering_distribution.png', dpi=150)
print("\nSaved: clustering_distribution.png")
plt.close()
print()

# ============================================================================
# 6. PATH LENGTHS AND SMALL-WORLD PROPERTY
# ============================================================================
print("STEP 6: Computing path lengths")
print("-"*70)

if nx.is_connected(G_main):
    print("Computing path lengths...")
    print("(This may take a minute for large networks)")

    L = nx.average_shortest_path_length(G_main)
    diameter = nx.diameter(G_main)

    # Expected path length for random network
    L_random = np.log(G_main.number_of_nodes()) / np.log(np.mean(degree_values))

    print(f"\nPath Length Statistics:")
    print(f"  Average shortest path L:  {L:.3f}")
    print(f"  Diameter (max path):      {diameter}")
    print(f"  Expected for random:      {L_random:.3f}")
    print(f"  Network size N:           {G_main.number_of_nodes():,}")
    print(f"  log(N):                   {np.log(G_main.number_of_nodes()):.3f}")

    # Small-world check
    print(f"\n  Small-world check:")
    print(f"    L ~ log(N)? {L:.2f} vs {np.log(G_main.number_of_nodes()):.2f}")
    if L < 2 * np.log(G_main.number_of_nodes()):
        print(f"    -> YES - Short path length (small-world property)")

    if C > 2 * C_random and L < 2 * np.log(G_main.number_of_nodes()):
        print(f"\n  SMALL-WORLD NETWORK: High clustering + Short paths")
else:
    print("Network is disconnected - path metrics computed on largest component")
    print(f"Largest component size: {G_main.number_of_nodes()} / {G.number_of_nodes()} nodes")
    L = None
    diameter = None

print()

# ============================================================================
# 7. CENTRALITY MEASURES
# ============================================================================
print("STEP 7: Computing centrality measures")
print("-"*70)
print("Computing centrality measures...")
print("(Betweenness may take a minute)")
print()

# Degree centrality (fast)
degree_cent = nx.degree_centrality(G)

# Betweenness centrality (slower)
betweenness = nx.betweenness_centrality(G)

# Closeness centrality (fast if connected)
closeness = nx.closeness_centrality(G)

# PageRank (fast)
pagerank = nx.pagerank(G)

print("Centrality computed!")

# Find top users by each metric
top_n = 10

print(f"\nTop {top_n} Users by Different Centrality Measures:")
print("="*70)

print(f"\nDegree Centrality (Most connected):")
top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
for i, (user, score) in enumerate(top_degree, 1):
    print(f"  {i:2d}. User {user}: {score:.4f} (degree = {degrees[user]})")

print(f"\nBetweenness Centrality (Best bridges):")
top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
for i, (user, score) in enumerate(top_between, 1):
    print(f"  {i:2d}. User {user}: {score:.4f}")

print(f"\nPageRank (Most influential):")
top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
for i, (user, score) in enumerate(top_pagerank, 1):
    print(f"  {i:2d}. User {user}: {score:.4f}")

# Visualize centrality distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(list(degree_cent.values()), bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Degree Centrality')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Degree Centrality Distribution')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(list(betweenness.values()), bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Betweenness Centrality')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Betweenness Centrality Distribution')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].hist(list(closeness.values()), bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Closeness Centrality')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Closeness Centrality Distribution')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(list(pagerank.values()), bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('PageRank')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('PageRank Distribution')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('centrality_distributions.png', dpi=150)
print("\nSaved: centrality_distributions.png")
plt.close()
print()

# ============================================================================
# 8. COMMUNITY DETECTION
# ============================================================================
print("STEP 8: Detecting communities")
print("-"*70)
print("Detecting communities using Louvain algorithm...")
print()

from networkx.algorithms import community as nx_community

# Louvain community detection
communities = nx_community.louvain_communities(G)

# Compute modularity
modularity = nx_community.modularity(G, communities)

print("Community Detection Results:")
print(f"  Number of communities: {len(communities)}")
print(f"  Modularity Q:          {modularity:.4f}")

if modularity > 0.7:
    print(f"  -> Very strong community structure")
elif modularity > 0.3:
    print(f"  -> Good community structure")
else:
    print(f"  -> Weak community structure")

# Analyze community sizes
community_sizes = [len(c) for c in communities]
community_sizes.sort(reverse=True)

print(f"\nCommunity sizes:")
print(f"  Largest:  {community_sizes[0]} nodes")
print(f"  Smallest: {community_sizes[-1]} nodes")
print(f"  Mean:     {np.mean(community_sizes):.1f} nodes")
print(f"  Median:   {np.median(community_sizes):.0f} nodes")

print(f"\nTop 10 largest communities:")
for i, size in enumerate(community_sizes[:10], 1):
    print(f"  {i:2d}. {size:4d} nodes ({100*size/G.number_of_nodes():.1f}%)")

# Visualize community size distribution
plt.figure(figsize=(10, 5))
plt.bar(range(len(community_sizes)), community_sizes, alpha=0.7)
plt.xlabel('Community (ranked by size)')
plt.ylabel('Number of nodes')
plt.title(f'Community Sizes (Modularity Q = {modularity:.3f})')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('community_sizes.png', dpi=150)
print("\nSaved: community_sizes.png")
plt.close()
print()

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("="*70)
print("NETWORK CHARACTERIZATION SUMMARY")
print("="*70)

print(f"\nBasic Properties:")
print(f"  Nodes (N):            {G.number_of_nodes():,}")
print(f"  Edges (E):            {G.number_of_edges():,}")
print(f"  Average degree ⟨k⟩:   {np.mean(degree_values):.2f}")
print(f"  Density:              {nx.density(G):.4f}")

print(f"\nStructural Properties:")
print(f"  Clustering C:         {C:.4f} (random: {C_random:.4f}, ratio: {C/C_random:.1f}x)")
if L is not None:
    print(f"  Avg path length L:    {L:.3f} (compare to log(N) = {np.log(G.number_of_nodes()):.2f})")
    print(f"  Diameter:             {diameter}")

print(f"\nCommunity Structure:")
print(f"  Communities:          {len(communities)}")
print(f"  Modularity Q:         {modularity:.4f}")

print(f"\nNetwork Type Assessment:")
print(f"  -> Check degree_distribution.png (log-log plot):")
print(f"      Linear trend = Scale-free (power-law)")
print(f"      Bell-shaped = Random-like")

if C > 2 * C_random:
    print(f"  -> High clustering -> Not random")

if L is not None and L < 2 * np.log(G.number_of_nodes()):
    print(f"  -> Short paths -> Small-world property")

if C > 2 * C_random and (L is not None and L < 2 * np.log(G.number_of_nodes())):
    print(f"\n  LIKELY A SMALL-WORLD NETWORK")
    print(f"     (High clustering + Short paths)")

if modularity > 0.3:
    print(f"  -> Strong communities detected")

print("\n" + "="*70)
print("Analysis complete!")
print(f"\nGenerated files:")
print(f"  - similarity_distribution.png")
print(f"  - degree_distribution.png")
print(f"  - clustering_distribution.png")
print(f"  - centrality_distributions.png")
print(f"  - community_sizes.png")
print("="*70)
