"""
Generate plots and summary from existing simulation results.
Run this after run_overnight.py to create missing visualization files.
"""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

RESULTS_DIR = 'results'

print("Loading results...")
si_df = pd.read_csv(f'{RESULTS_DIR}/big_matrix_si_influence.csv', index_col='user_id')
sir_df = pd.read_csv(f'{RESULTS_DIR}/big_matrix_sir_influence.csv', index_col='user_id')

print(f"SI results: {len(si_df)} nodes")
print(f"SIR results: {len(sir_df)} nodes")

# Correlations
si_corr_strength = si_df['time_to_50'].corr(si_df['strength'])
si_corr_pagerank = si_df['time_to_50'].corr(si_df['pagerank'])
sir_corr_strength = sir_df['final_outbreak'].corr(sir_df['strength'])
sir_corr_pagerank = sir_df['final_outbreak'].corr(sir_df['pagerank'])

# Summary text (using ASCII instead of Greek letters to avoid encoding issues)
summary_text = f"""
================================================================================
OVERNIGHT SIMULATION RESULTS - BIG MATRIX
================================================================================
Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PARAMETERS:
  Beta:               0.3
  Gamma:              0.1
  Similarity thresh:  0.3
  Runs per node:      10

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

  Correlations:
    Strength:         {sir_corr_strength:.4f}
    PageRank:         {sir_corr_pagerank:.4f}

  Top 10 largest outbreaks:
{sir_df.head(10)[['final_outbreak', 'attack_rate', 'strength', 'pagerank']].to_string()}

FILES SAVED:
  - results/big_matrix_si_influence.csv
  - results/big_matrix_sir_influence.csv
  - results/big_matrix_summary.txt
  - results/big_matrix_plots.png
  - results/big_matrix_log.txt

================================================================================
"""

print("\nWriting summary...")
with open(f'{RESULTS_DIR}/big_matrix_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"Summary saved to {RESULTS_DIR}/big_matrix_summary.txt")

# Generate plots
print("\nGenerating plots...")
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
plt.savefig(f'{RESULTS_DIR}/big_matrix_plots.png', dpi=150, bbox_inches='tight')
print(f"Plots saved to {RESULTS_DIR}/big_matrix_plots.png")

print("\n✓ Done! All files generated successfully.")
print(summary_text)
