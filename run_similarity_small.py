"""
Overnight SI and SIR simulation on SMALL MATRIX dataset.
Run on remote PC.

Usage:
    python run_similarity_small.py

Results saved to:
    results/small_matrix_si_influence.csv
    results/small_matrix_sir_influence.csv
    results/small_matrix_summary.txt
    results/small_matrix_plots.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for remote
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_FILE = 'data/raw/small_matrix.csv'  # SMALL matrix
# Create timestamped results directory
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = f'results/run_{TIMESTAMP}'
BETA = 0.05  # Changed from 0.3
GAMMA = 0.15  # Changed from 0.1
NUM_RUNS_PER_NODE = 20  # Runs per node for statistics
MAX_STEPS_SI = 200
MAX_STEPS_SIR = 300
SIMILARITY_THRESHOLD = 0.7  # Only connect users with similarity >= this
SAMPLE_SIZE = None  # Number of nodes to simulate (set to None for all nodes)
NUM_PROCESSES = max(1, cpu_count() - 1)  # Use all cores except one

os.makedirs(RESULTS_DIR, exist_ok=True)

# Logging
log_file = open(f'{RESULTS_DIR}/small_matrix_log.txt', 'w', encoding='utf-8')
def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    log_file.write(full_msg + '\n')
    log_file.flush()

# ============================================================================
# SIMULATION FUNCTIONS (defined at module level for multiprocessing)
# ============================================================================

def run_si(G, seed, beta, max_steps):
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

def measure_si_influence(G, seed, beta, num_runs, max_steps):
    threshold_50 = G.number_of_nodes() * 0.5
    threshold_90 = G.number_of_nodes() * 0.9

    times_to_50 = []
    times_to_90 = []
    final_sizes = []

    for _ in range(num_runs):
        counts = run_si(G, seed, beta, max_steps)

        t50 = next((t for t, c in enumerate(counts) if c >= threshold_50), len(counts))
        times_to_50.append(t50)

        t90 = next((t for t, c in enumerate(counts) if c >= threshold_90), len(counts))
        times_to_90.append(t90)

        final_sizes.append(counts[-1])

    return {
        'time_to_50': np.mean(times_to_50),
        'time_to_90': np.mean(times_to_90),
        'std_time_50': np.std(times_to_50),
        'final_size': np.mean(final_sizes),
    }

def process_si_node(node, G, strength, pagerank, beta, num_runs, max_steps):
    """Wrapper function for parallel processing"""
    result = measure_si_influence(G, node, beta, num_runs, max_steps)
    result['strength'] = strength[node]
    result['pagerank'] = pagerank[node]
    return node, result

def run_sir(G, seed, beta, gamma, max_steps):
    """Run SIR simulation, return S, I, R counts over time."""
    infected = {seed}
    recovered = set()

    S_counts = [G.number_of_nodes() - 1]
    I_counts = [1]
    R_counts = [0]

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

        # Track
        S_counts.append(G.number_of_nodes() - len(infected) - len(recovered))
        I_counts.append(len(infected))
        R_counts.append(len(recovered))

        if len(infected) == 0:
            break

    return S_counts, I_counts, R_counts

def measure_sir_influence(G, seed, beta, gamma, num_runs, max_steps):
    """Measure extended influence metrics for a seed node."""
    final_outbreak_sizes = []
    peak_infections = []
    epidemic_durations = []
    times_to_peak = []
    R0_values = []
    early_growth_rates = []

    threshold_epidemic = G.number_of_nodes() * 0.1  # 10% of network

    for _ in range(num_runs):
        S, I, R = run_sir(G, seed, beta, gamma, max_steps)

        # Existing metrics
        final_outbreak_sizes.append(R[-1])
        peak_infections.append(max(I))
        epidemic_durations.append(len(I))
        times_to_peak.append(I.index(max(I)))

        # R0 (new infections in first step)
        R0_values.append(I[1] - I[0] if len(I) > 1 else 0)

        # Early growth rate (exponential fit to first 5 steps)
        if len(I) >= 5:
            early_I = I[1:6]
            if all(x > 0 for x in early_I):
                try:
                    growth_rate = np.polyfit(range(5), np.log(early_I), 1)[0]
                    early_growth_rates.append(growth_rate)
                except:
                    pass  # Skip if polyfit fails

    return {
        'final_outbreak': np.mean(final_outbreak_sizes),
        'std_outbreak': np.std(final_outbreak_sizes),
        'attack_rate': np.mean(final_outbreak_sizes) / G.number_of_nodes(),
        'epidemic_probability': sum(1 for x in final_outbreak_sizes if x > threshold_epidemic) / num_runs,
        'peak_infection': np.mean(peak_infections),
        'time_to_peak': np.mean(times_to_peak),
        'duration': np.mean(epidemic_durations),
        'R0': np.mean(R0_values),
        'growth_rate': np.mean(early_growth_rates) if early_growth_rates else 0.0,
    }

def process_sir_node(node, G, strength, pagerank, beta, gamma, num_runs, max_steps):
    """Wrapper function for parallel processing"""
    result = measure_sir_influence(G, node, beta, gamma, num_runs, max_steps)
    result['strength'] = strength[node]
    result['pagerank'] = pagerank[node]
    return node, result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    log("="*70)
    log("OVERNIGHT SI/SIR SIMULATION - SMALL MATRIX")
    log("="*70)
    log(f"Data file: {DATA_FILE}")
    log(f"Parameters: β={BETA}, γ={GAMMA}, runs={NUM_RUNS_PER_NODE}")
    log(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    log(f"Parallel processes: {NUM_PROCESSES} (out of {cpu_count()} CPUs)")
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*70)

    # ============================================================================
    # LOAD NETWORK
    # ============================================================================
    start_time = time.time()
    log("\n1. Loading data and building network...")

    try:
        interactions = pd.read_csv(DATA_FILE)
        log(f"   Loaded {len(interactions):,} interactions")
    except Exception as e:
        log(f"   ERROR loading data: {e}")
        log("   Exiting.")
        exit(1)

    log(f"   Unique users: {interactions['user_id'].nunique():,}")
    log(f"   Unique videos: {interactions['video_id'].nunique():,}")

    log("   Creating user-video matrix...")
    user_video_matrix = interactions.pivot_table(
        index='user_id',
        columns='video_id',
        values='watch_ratio',
        fill_value=0
    )
    log(f"   Matrix shape: {user_video_matrix.shape[0]:,} × {user_video_matrix.shape[1]:,}")

    log("   Computing similarity matrix...")
    similarity_matrix = cosine_similarity(user_video_matrix.values)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_video_matrix.index,
        columns=user_video_matrix.index
    )

    log("   Building graph...")
    G = nx.Graph()
    users = user_video_matrix.index.tolist()
    G.add_nodes_from(users)

    edge_count = 0
    for i, user_i in enumerate(users):
        if i % 100 == 0:
            log(f"   Adding edges: {i}/{len(users)} users processed ({edge_count:,} edges so far)")
        for j, user_j in enumerate(users[i+1:], start=i+1):
            sim = similarity_df.loc[user_i, user_j]
            if sim >= SIMILARITY_THRESHOLD:  # Only add edge if similarity is high enough
                G.add_edge(user_i, user_j, weight=sim)
                edge_count += 1

    mean_weight = np.mean([d['weight'] for u, v, d in G.edges(data=True)])
    log(f"   ✓ Network built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    log(f"   Mean similarity: {mean_weight:.4f}")
    log(f"   Density: {nx.density(G):.4f}")
    log(f"   Time: {time.time() - start_time:.1f}s")

    # Compute centrality
    log("\n2. Computing centrality measures...")
    start_time = time.time()
    strength = dict(G.degree(weight='weight'))
    pagerank = nx.pagerank(G, weight='weight')
    log(f"   ✓ Centrality computed")
    log(f"   Time: {time.time() - start_time:.1f}s")

    # Sample nodes if requested
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(users):
        log(f"\n   Sampling {SAMPLE_SIZE} nodes from {len(users)} total nodes...")
        np.random.seed(42)  # For reproducibility
        sampled_users = np.random.choice(users, size=SAMPLE_SIZE, replace=False).tolist()
        log(f"   ✓ Sampled {len(sampled_users)} nodes")
    else:
        sampled_users = users
        log(f"\n   Using all {len(users)} nodes (no sampling)")

    # ============================================================================
    # SI MODEL
    # ============================================================================
    log("\n" + "="*70)
    log("3. RUNNING SI MODEL")
    log("="*70)

    start_time = time.time()
    log(f"   Running {len(sampled_users)} nodes × {NUM_RUNS_PER_NODE} simulations...")
    log(f"   Using {NUM_PROCESSES} parallel processes")

    # Create partial function with fixed parameters
    process_func = partial(process_si_node, G=G, strength=strength, pagerank=pagerank,
                           beta=BETA, num_runs=NUM_RUNS_PER_NODE, max_steps=MAX_STEPS_SI)

    # Run in parallel
    si_influence = {}
    with Pool(processes=NUM_PROCESSES) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_func, sampled_users),
            total=len(sampled_users),
            desc="SI Progress"
        ))

        # Convert to dictionary
        for node, result in results:
            si_influence[node] = result

        # Log progress
        elapsed = time.time() - start_time
        log(f"   ✓ Completed {len(sampled_users)} nodes in {elapsed/60:.1f} minutes")

    si_df = pd.DataFrame.from_dict(si_influence, orient='index')
    si_df.index.name = 'user_id'
    si_df = si_df.sort_values('time_to_50')

    si_df.to_csv(f'{RESULTS_DIR}/small_matrix_si_influence.csv')
    log(f"   ✓ SI results saved: {RESULTS_DIR}/small_matrix_si_influence.csv")
    log(f"   Total time: {(time.time() - start_time)/60:.1f} minutes")

    # ============================================================================
    # SIR MODEL
    # ============================================================================
    log("\n" + "="*70)
    log("4. RUNNING SIR MODEL")
    log("="*70)

    start_time = time.time()
    log(f"   Running {len(sampled_users)} nodes × {NUM_RUNS_PER_NODE} simulations...")
    log(f"   Using {NUM_PROCESSES} parallel processes")

    # Create partial function with fixed parameters
    process_func = partial(process_sir_node, G=G, strength=strength, pagerank=pagerank,
                           beta=BETA, gamma=GAMMA, num_runs=NUM_RUNS_PER_NODE, max_steps=MAX_STEPS_SIR)

    # Run in parallel
    sir_influence = {}
    with Pool(processes=NUM_PROCESSES) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_func, sampled_users),
            total=len(sampled_users),
            desc="SIR Progress"
        ))

        # Convert to dictionary
        for node, result in results:
            sir_influence[node] = result

        # Log progress
        elapsed = time.time() - start_time
        log(f"   ✓ Completed {len(sampled_users)} nodes in {elapsed/60:.1f} minutes")

    sir_df = pd.DataFrame.from_dict(sir_influence, orient='index')
    sir_df.index.name = 'user_id'
    sir_df = sir_df.sort_values('final_outbreak', ascending=False)

    sir_df.to_csv(f'{RESULTS_DIR}/small_matrix_sir_influence.csv')
    log(f"   ✓ SIR results saved: {RESULTS_DIR}/small_matrix_sir_influence.csv")
    log(f"   Total time: {(time.time() - start_time)/60:.1f} minutes")

    # ============================================================================
    # ANALYSIS AND SUMMARY
    # ============================================================================
    log("\n" + "="*70)
    log("5. GENERATING SUMMARY AND PLOTS")
    log("="*70)

    # Correlations
    si_corr_strength = si_df['time_to_50'].corr(si_df['strength'])
    si_corr_pagerank = si_df['time_to_50'].corr(si_df['pagerank'])
    sir_corr_strength = sir_df['final_outbreak'].corr(sir_df['strength'])
    sir_corr_pagerank = sir_df['final_outbreak'].corr(sir_df['pagerank'])

    # Summary text
    summary_text = f"""
    ================================================================================
    OVERNIGHT SIMULATION RESULTS - SMALL MATRIX
    ================================================================================
    Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    NETWORK PROPERTIES:
      Nodes:              {G.number_of_nodes():,}
      Edges:              {G.number_of_edges():,}
      Density:            {nx.density(G):.4f}
      Mean similarity:    {mean_weight:.4f}

    PARAMETERS:
      Beta (β):           {BETA}
      Gamma (γ):          {GAMMA}
      Similarity thresh:  {SIMILARITY_THRESHOLD}
      Runs per node:      {NUM_RUNS_PER_NODE}

    SI MODEL RESULTS:
      Metric:             Time to 50% infection
      Fastest spreader:   User {si_df.index[0]} (t50={si_df.iloc[0]['time_to_50']:.2f})
      Slowest spreader:   User {si_df.index[-1]} (t50={si_df.iloc[-1]['time_to_50']:.2f})

      Correlations:
        Strength:         {si_corr_strength:.4f} (negative = faster spreading)
        PageRank:         {si_corr_pagerank:.4f}

      Top 10 fastest spreaders:
    {si_df.head(10)[['time_to_50', 'strength', 'pagerank']].to_string()}

    SIR MODEL RESULTS:
      Metric:             Final outbreak size
      Largest outbreak:   User {sir_df.index[0]} (size={sir_df.iloc[0]['final_outbreak']:.1f}, rate={sir_df.iloc[0]['attack_rate']*100:.1f}%)
      Smallest outbreak:  User {sir_df.index[-1]} (size={sir_df.iloc[-1]['final_outbreak']:.1f}, rate={sir_df.iloc[-1]['attack_rate']*100:.1f}%)
      Mean attack rate:   {sir_df['attack_rate'].mean()*100:.2f}%
      Mean R0:            {sir_df['R0'].mean():.2f}
      Mean epidemic prob: {sir_df['epidemic_probability'].mean()*100:.1f}%

      Correlations:
        Strength:         {sir_corr_strength:.4f}
        PageRank:         {sir_corr_pagerank:.4f}

      Top 10 largest outbreaks:
    {sir_df.head(10)[['final_outbreak', 'attack_rate', 'R0', 'epidemic_probability', 'strength']].to_string()}

    FILES SAVED:
      - {RESULTS_DIR}/small_matrix_si_influence.csv
      - {RESULTS_DIR}/small_matrix_sir_influence.csv
      - {RESULTS_DIR}/small_matrix_summary.txt
      - {RESULTS_DIR}/small_matrix_plots.png
      - {RESULTS_DIR}/small_matrix_log.txt

    ================================================================================
    """

    with open(f'{RESULTS_DIR}/small_matrix_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)

    log("\n" + summary_text)

    # Generate plots
    log("\n   Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # SI: Time to 50% vs Strength
    axes[0, 0].scatter(si_df['strength'], si_df['time_to_50'], alpha=0.4, s=20)
    axes[0, 0].set_xlabel('Strength')
    axes[0, 0].set_ylabel('Time to 50% Infection')
    axes[0, 0].set_title(f'SI: Spreading Speed vs Strength (r={si_corr_strength:.3f})')
    axes[0, 0].grid(alpha=0.3)

    # SI: Distribution of times
    axes[0, 1].hist(si_df['time_to_50'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Time to 50% Infection')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'SI: Distribution of Spreading Times')
    axes[0, 1].grid(alpha=0.3)

    # SIR: Outbreak size vs Strength
    axes[1, 0].scatter(sir_df['strength'], sir_df['final_outbreak'], alpha=0.4, s=20)
    axes[1, 0].set_xlabel('Strength')
    axes[1, 0].set_ylabel('Final Outbreak Size')
    axes[1, 0].set_title(f'SIR: Outbreak Size vs Strength (r={sir_corr_strength:.3f})')
    axes[1, 0].grid(alpha=0.3)

    # SIR: Distribution of attack rates
    axes[1, 1].hist(sir_df['attack_rate'] * 100, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Attack Rate (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'SIR: Distribution of Attack Rates (mean={sir_df["attack_rate"].mean()*100:.1f}%)')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/small_matrix_plots.png', dpi=150, bbox_inches='tight')
    log(f"   ✓ Plots saved: {RESULTS_DIR}/small_matrix_plots.png")

    # ============================================================================
    # COMPLETION
    # ============================================================================
    log("\n" + "="*70)
    log("✓✓✓ ALL SIMULATIONS COMPLETED SUCCESSFULLY ✓✓✓")
    log("="*70)
    log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("\nCheck results in results/ folder:")
    log("  - small_matrix_si_influence.csv")
    log("  - small_matrix_sir_influence.csv")
    log("  - small_matrix_summary.txt")
    log("  - small_matrix_plots.png")
    log("="*70)

    log_file.close()


# ============================================================================
# POST-RUN ANALYSIS FUNCTION
# ============================================================================

def analyze_existing_results(results_dir='results'):
    """
    Analyze existing CSV results and generate summary + plots.
    Usage: python run_similarity_small.py --analyze results_run2
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime

    print(f"\n{'='*70}")
    print(f"ANALYZING RESULTS FROM: {results_dir}")
    print(f"{'='*70}\n")

    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"ERROR: Directory '{results_dir}' not found!")
        return

    # Load CSV files
    si_file = f'{results_dir}/small_matrix_si_influence.csv'
    sir_file = f'{results_dir}/small_matrix_sir_influence.csv'

    if not os.path.exists(si_file) or not os.path.exists(sir_file):
        print(f"ERROR: CSV files not found in '{results_dir}'!")
        print(f"  Looking for: {si_file}")
        print(f"  Looking for: {sir_file}")
        return

    print(f"Loading data files...")
    si_df = pd.read_csv(si_file)
    sir_df = pd.read_csv(sir_file)
    print(f"  ✓ SI data: {len(si_df)} rows")
    print(f"  ✓ SIR data: {len(sir_df)} rows")

    # Generate summary statistics
    print(f"\nGenerating summary statistics...")

    # SI correlations
    si_corr_strength = si_df['strength'].corr(si_df['time_to_50'])
    si_corr_degree = si_df['degree'].corr(si_df['time_to_50'])
    si_corr_betweenness = si_df['betweenness'].corr(si_df['time_to_50'])
    si_corr_closeness = si_df['closeness'].corr(si_df['time_to_50'])

    # SIR correlations
    sir_corr_strength = sir_df['strength'].corr(sir_df['final_outbreak'])
    sir_corr_degree = sir_df['degree'].corr(sir_df['final_outbreak'])
    sir_corr_betweenness = sir_df['betweenness'].corr(sir_df['final_outbreak'])
    sir_corr_closeness = sir_df['closeness'].corr(sir_df['final_outbreak'])

    sir_corr_r0_strength = sir_df['strength'].corr(sir_df['R0'])
    sir_corr_r0_degree = sir_df['degree'].corr(sir_df['R0'])

    # Build summary text
    summary_text = f"""
    ================================================================================
                          SIMILARITY NETWORK INFLUENCE ANALYSIS
                                   (Post-Run Analysis)
    ================================================================================

    ANALYSIS DIRECTORY: {results_dir}
    ANALYSIS TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ================================================================================
    SI MODEL RESULTS (Spreading Speed - Time to 50% Infection)
    ================================================================================

    Correlations with Time to 50% (negative = faster spread):
      • Strength:      {si_corr_strength:7.4f}
      • Degree:        {si_corr_degree:7.4f}
      • Betweenness:   {si_corr_betweenness:7.4f}
      • Closeness:     {si_corr_closeness:7.4f}

    Distribution Statistics:
      • Mean time to 50%:     {si_df['time_to_50'].mean():.2f} steps
      • Std time to 50%:      {si_df['time_to_50'].std():.2f} steps
      • Median time to 50%:   {si_df['time_to_50'].median():.2f} steps
      • Min time to 50%:      {si_df['time_to_50'].min():.2f} steps
      • Max time to 50%:      {si_df['time_to_50'].max():.2f} steps

    ================================================================================
    SIR MODEL RESULTS (Epidemic Size - Final Outbreak)
    ================================================================================

    Correlations with Final Outbreak Size:
      • Strength:      {sir_corr_strength:7.4f}
      • Degree:        {sir_corr_degree:7.4f}
      • Betweenness:   {sir_corr_betweenness:7.4f}
      • Closeness:     {sir_corr_closeness:7.4f}

    Correlations with R0:
      • Strength:      {sir_corr_r0_strength:7.4f}
      • Degree:        {sir_corr_r0_degree:7.4f}

    Outbreak Statistics:
      • Mean final outbreak:  {sir_df['final_outbreak'].mean():.2f} nodes
      • Mean attack rate:     {sir_df['attack_rate'].mean()*100:.2f}%
      • Mean epidemic prob:   {sir_df['epidemic_probability'].mean()*100:.2f}%
      • Mean R0:              {sir_df['R0'].mean():.3f}
      • Mean peak infection:  {sir_df['peak_infection'].mean():.2f} nodes
      • Mean time to peak:    {sir_df['time_to_peak'].mean():.2f} steps
      • Mean duration:        {sir_df['duration'].mean():.2f} steps

    ================================================================================
    KEY FINDINGS
    ================================================================================

    Most influential metric for SI spreading speed:
      → {'Strength' if abs(si_corr_strength) == max(abs(si_corr_strength), abs(si_corr_degree), abs(si_corr_betweenness), abs(si_corr_closeness)) else 'Degree' if abs(si_corr_degree) == max(abs(si_corr_strength), abs(si_corr_degree), abs(si_corr_betweenness), abs(si_corr_closeness)) else 'Betweenness' if abs(si_corr_betweenness) == max(abs(si_corr_strength), abs(si_corr_degree), abs(si_corr_betweenness), abs(si_corr_closeness)) else 'Closeness'} (r = {max(abs(si_corr_strength), abs(si_corr_degree), abs(si_corr_betweenness), abs(si_corr_closeness)):.4f})

    Most influential metric for SIR outbreak size:
      → {'Strength' if abs(sir_corr_strength) == max(abs(sir_corr_strength), abs(sir_corr_degree), abs(sir_corr_betweenness), abs(sir_corr_closeness)) else 'Degree' if abs(sir_corr_degree) == max(abs(sir_corr_strength), abs(sir_corr_degree), abs(sir_corr_betweenness), abs(sir_corr_closeness)) else 'Betweenness' if abs(sir_corr_betweenness) == max(abs(sir_corr_strength), abs(sir_corr_degree), abs(sir_corr_betweenness), abs(sir_corr_closeness)) else 'Closeness'} (r = {max(abs(sir_corr_strength), abs(sir_corr_degree), abs(sir_corr_betweenness), abs(sir_corr_closeness)):.4f})

    FILES SAVED:
      - {results_dir}/small_matrix_summary.txt
      - {results_dir}/small_matrix_plots.png

    ================================================================================
    """

    # Save summary
    print(f"\nSaving summary to: {results_dir}/small_matrix_summary.txt")
    with open(f'{results_dir}/small_matrix_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"  ✓ Summary saved")

    # Generate plots
    print(f"\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # SI: Time to 50% vs Strength
    axes[0, 0].scatter(si_df['strength'], si_df['time_to_50'], alpha=0.4, s=20)
    axes[0, 0].set_xlabel('Strength')
    axes[0, 0].set_ylabel('Time to 50% Infection')
    axes[0, 0].set_title(f'SI: Spreading Speed vs Strength (r={si_corr_strength:.3f})')
    axes[0, 0].grid(alpha=0.3)

    # SI: Distribution of times
    axes[0, 1].hist(si_df['time_to_50'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Time to 50% Infection')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'SI: Distribution of Spreading Times (mean={si_df["time_to_50"].mean():.1f})')
    axes[0, 1].grid(alpha=0.3)

    # SIR: Final outbreak vs Strength
    axes[1, 0].scatter(sir_df['strength'], sir_df['final_outbreak'], alpha=0.4, s=20)
    axes[1, 0].set_xlabel('Strength')
    axes[1, 0].set_ylabel('Final Outbreak Size')
    axes[1, 0].set_title(f'SIR: Outbreak Size vs Strength (r={sir_corr_strength:.3f})')
    axes[1, 0].grid(alpha=0.3)

    # SIR: Distribution of attack rates
    axes[1, 1].hist(sir_df['attack_rate'] * 100, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Attack Rate (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'SIR: Distribution of Attack Rates (mean={sir_df["attack_rate"].mean()*100:.1f}%)')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/small_matrix_plots.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Plots saved: {results_dir}/small_matrix_plots.png")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(summary_text)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    # Check for --analyze flag
    if len(sys.argv) > 1 and sys.argv[1] == '--analyze':
        if len(sys.argv) > 2:
            analyze_existing_results(sys.argv[2])
        else:
            print("Usage: python run_similarity_small.py --analyze <results_directory>")
            print("Example: python run_similarity_small.py --analyze results_run2")
    else:
        # Run normal execution (the code above)
        pass
