"""
Run both SI and SIR simulations as a script (slightly faster than notebooks).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

print("="*70)
print("RUNNING SI AND SIR SIMULATIONS")
print("="*70)

# Load network
print("\nLoading network...")
interactions = pd.read_csv('../data/raw/small_matrix.csv')

user_video_matrix = interactions.pivot_table(
    index='user_id',
    columns='video_id',
    values='watch_ratio',
    fill_value=0
)

similarity_matrix = cosine_similarity(user_video_matrix.values)
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=user_video_matrix.index,
    columns=user_video_matrix.index
)

G = nx.Graph()
users = user_video_matrix.index.tolist()
G.add_nodes_from(users)

for i, user_i in enumerate(users):
    for j, user_j in enumerate(users[i+1:], start=i+1):
        sim = similarity_df.loc[user_i, user_j]
        G.add_edge(user_i, user_j, weight=sim)

print(f"Network loaded: {G.number_of_nodes():,} nodes")

# Compute centrality
strength = dict(G.degree(weight='weight'))
pagerank = nx.pagerank(G, weight='weight')

# Parameters
BETA = 0.3
GAMMA = 0.1
NUM_RUNS = 20

# ============================================================================
# SI MODEL
# ============================================================================
print("\n" + "="*70)
print("RUNNING SI MODEL")
print("="*70)

def run_si(G, seed, beta, max_steps=50):
    infected = {seed}
    counts = [1]

    for t in range(max_steps):
        new_infected = set()
        for node in infected:
            for neighbor in G.neighbors(node):
                if neighbor not in infected:
                    if np.random.random() < beta * G[node][neighbor]['weight']:
                        new_infected.add(neighbor)

        if not new_infected:
            break
        infected.update(new_infected)
        counts.append(len(infected))

    return counts

def measure_si_influence(G, seed, beta, num_runs):
    threshold_50 = G.number_of_nodes() * 0.5
    times_to_50 = []

    for _ in range(num_runs):
        counts = run_si(G, seed, beta)
        t50 = next((t for t, c in enumerate(counts) if c >= threshold_50), len(counts))
        times_to_50.append(t50)

    return np.mean(times_to_50)

print(f"\nMeasuring SI influence (β={BETA}, {NUM_RUNS} runs per node)...")
si_influence = {}
for node in tqdm(users, desc="SI"):
    si_influence[node] = {
        'time_to_50': measure_si_influence(G, node, BETA, NUM_RUNS),
        'strength': strength[node],
        'pagerank': pagerank[node],
    }

si_df = pd.DataFrame.from_dict(si_influence, orient='index')
si_df.index.name = 'user_id'
si_df = si_df.sort_values('time_to_50')

os.makedirs('../results', exist_ok=True)
si_df.to_csv('../results/similarity_network_si_influence.csv')
print("✓ SI results saved")

# ============================================================================
# SIR MODEL
# ============================================================================
print("\n" + "="*70)
print("RUNNING SIR MODEL")
print("="*70)

def run_sir(G, seed, beta, gamma, max_steps=100):
    infected = {seed}
    recovered = set()

    for t in range(max_steps):
        # Recovery
        newly_recovered = set()
        for node in infected:
            if np.random.random() < gamma:
                newly_recovered.add(node)

        infected -= newly_recovered
        recovered.update(newly_recovered)

        # Infection
        new_infected = set()
        for node in infected:
            for neighbor in G.neighbors(node):
                if neighbor not in infected and neighbor not in recovered:
                    if np.random.random() < beta * G[node][neighbor]['weight']:
                        new_infected.add(neighbor)

        infected.update(new_infected)

        if len(infected) == 0:
            break

    return len(recovered)

def measure_sir_influence(G, seed, beta, gamma, num_runs):
    outbreaks = []
    for _ in range(num_runs):
        outbreak = run_sir(G, seed, beta, gamma)
        outbreaks.append(outbreak)
    return np.mean(outbreaks)

print(f"\nMeasuring SIR influence (β={BETA}, γ={GAMMA}, {NUM_RUNS} runs per node)...")
sir_influence = {}
for node in tqdm(users, desc="SIR"):
    sir_influence[node] = {
        'final_outbreak': measure_sir_influence(G, node, BETA, GAMMA, NUM_RUNS),
        'strength': strength[node],
        'pagerank': pagerank[node],
    }

sir_df = pd.DataFrame.from_dict(sir_influence, orient='index')
sir_df.index.name = 'user_id'
sir_df = sir_df.sort_values('final_outbreak', ascending=False)

sir_df.to_csv('../results/similarity_network_sir_influence.csv')
print("✓ SIR results saved")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nSI Model:")
print(f"  Fastest spreader: User {si_df.index[0]} (t50={si_df.iloc[0]['time_to_50']:.1f})")
print(f"  Correlation with strength: {si_df['time_to_50'].corr(si_df['strength']):.3f}")

print(f"\nSIR Model:")
print(f"  Largest outbreak: User {sir_df.index[0]} (size={sir_df.iloc[0]['final_outbreak']:.1f})")
print(f"  Correlation with strength: {sir_df['final_outbreak'].corr(sir_df['strength']):.3f}")

print("\n✓ All results saved to results/")
print("="*70)
