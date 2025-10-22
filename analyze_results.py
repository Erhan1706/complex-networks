"""
Analyze existing results from overnight runs.

Usage:
    python analyze_results.py results_run2
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def analyze_existing_results(results_dir='results'):
    """
    Analyze existing CSV results and generate summary + plots.
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING RESULTS FROM: {results_dir}")
    print(f"{'='*70}\n")

    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"ERROR: Directory '{results_dir}' not found!")
        return

    # Load CSV files
    si_file = f'{results_dir}/big_matrix_si_influence.csv'
    sir_file = f'{results_dir}/big_matrix_sir_influence.csv'

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
    si_corr_pagerank = si_df['pagerank'].corr(si_df['time_to_50'])

    # SIR correlations
    sir_corr_strength = sir_df['strength'].corr(sir_df['final_outbreak'])
    sir_corr_pagerank = sir_df['pagerank'].corr(sir_df['final_outbreak'])

    sir_corr_r0_strength = sir_df['strength'].corr(sir_df['R0'])
    sir_corr_r0_pagerank = sir_df['pagerank'].corr(sir_df['R0'])

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
      • PageRank:      {si_corr_pagerank:7.4f}

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
      • PageRank:      {sir_corr_pagerank:7.4f}

    Correlations with R0:
      • Strength:      {sir_corr_r0_strength:7.4f}
      • PageRank:      {sir_corr_r0_pagerank:7.4f}

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
      → {'Strength' if abs(si_corr_strength) > abs(si_corr_pagerank) else 'PageRank'} (r = {max(abs(si_corr_strength), abs(si_corr_pagerank)):.4f})

    Most influential metric for SIR outbreak size:
      → {'Strength' if abs(sir_corr_strength) > abs(sir_corr_pagerank) else 'PageRank'} (r = {max(abs(sir_corr_strength), abs(sir_corr_pagerank)):.4f})

    FILES SAVED:
      - {results_dir}/big_matrix_summary.txt
      - {results_dir}/big_matrix_plots.png

    ================================================================================
    """

    # Save summary
    print(f"\nSaving summary to: {results_dir}/big_matrix_summary.txt")
    with open(f'{results_dir}/big_matrix_summary.txt', 'w', encoding='utf-8') as f:
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
    plt.savefig(f'{results_dir}/big_matrix_plots.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Plots saved: {results_dir}/big_matrix_plots.png")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(summary_text)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_existing_results(sys.argv[1])
    else:
        print("Usage: python analyze_results.py <results_directory>")
        print("Example: python analyze_results.py results_run2")
