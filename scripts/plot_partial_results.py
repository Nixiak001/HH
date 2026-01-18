#!/usr/bin/env python
"""
Visualize partial experiment results during training.

This script can be used while experiments are still running to:
- Plot learning curves for completed and ongoing runs
- Show progress of each algorithm and seed
- Generate preliminary comparison plots
- Work with incomplete data (missing seeds, partial training)

Usage:
    python scripts/plot_partial_results.py --results_dir comparison_results/mamujoco_Humanoid-v2_17x1
"""

import argparse
import os
import json
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("ERROR: matplotlib/seaborn required. Install with: pip install matplotlib seaborn")
    sys.exit(1)

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("ERROR: tensorboard required. Install with: pip install tensorboard")
    sys.exit(1)


def find_tensorboard_logs(results_dir):
    """Find all TensorBoard event files in the results directory."""
    pattern = os.path.join(results_dir, "**", "events.out.tfevents.*")
    event_files = glob.glob(pattern, recursive=True)
    return event_files


def parse_tensorboard_log(event_file):
    """Parse a TensorBoard event file and extract scalar metrics."""
    try:
        ea = event_accumulator.EventAccumulator(
            os.path.dirname(event_file),
            size_guidance={
                event_accumulator.SCALARS: 0,
            }
        )
        ea.Reload()
        
        # Get all scalar tags
        tags = ea.Tags().get('scalars', [])
        
        data = {}
        for tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = {'steps': steps, 'values': values}
        
        return data
    except Exception as e:
        print(f"Warning: Could not parse {event_file}: {e}")
        return {}


def extract_algo_seed_from_path(path):
    """Extract algorithm name and seed from result path."""
    parts = path.split('/')
    
    # Look for directory or file with algo and seed info
    algo = None
    seed = None
    
    for part in reversed(parts):
        if 'happo' in part.lower():
            algo = 'happo'
        elif 'mappo' in part.lower():
            algo = 'mappo'
        elif 'hatd3' in part.lower():
            algo = 'hatd3'
        
        # Extract seed
        if 'seed' in part.lower():
            try:
                seed = int(part.split('seed')[-1].split('_')[0])
            except:
                pass
    
    return algo, seed


def aggregate_partial_results(results_dir):
    """Aggregate all available results, even if incomplete."""
    event_files = find_tensorboard_logs(results_dir)
    
    if not event_files:
        print(f"No TensorBoard event files found in {results_dir}")
        return None
    
    print(f"Found {len(event_files)} TensorBoard event files")
    
    # Organize data by algorithm and seed
    algo_data = defaultdict(lambda: defaultdict(dict))
    
    for event_file in event_files:
        algo, seed = extract_algo_seed_from_path(event_file)
        
        if algo is None:
            continue
        
        data = parse_tensorboard_log(event_file)
        
        if data:
            algo_data[algo][seed] = data
            print(f"  ✓ {algo} seed {seed}: {len(data)} metrics")
    
    return dict(algo_data)


def get_available_metrics(algo_data):
    """Get list of available metrics across all runs."""
    all_metrics = set()
    
    for algo, seed_data in algo_data.items():
        for seed, data in seed_data.items():
            all_metrics.update(data.keys())
    
    return sorted(list(all_metrics))


def plot_individual_runs(algo_data, metric_key, output_path):
    """Plot all individual runs for each algorithm."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'happo': '#1f77b4', 'mappo': '#ff7f0e', 'hatd3': '#2ca02c'}
    algo_names = {'happo': 'HAPPO', 'mappo': 'MAPPO', 'hatd3': 'HATD3'}
    
    for idx, algo in enumerate(['happo', 'mappo', 'hatd3']):
        ax = axes[idx]
        
        if algo not in algo_data:
            ax.text(0.5, 0.5, f'No data for {algo.upper()}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(algo_names[algo])
            continue
        
        seed_data = algo_data[algo]
        color = colors[algo]
        
        for seed, data in seed_data.items():
            if metric_key not in data:
                continue
            
            steps = np.array(data[metric_key]['steps'])
            values = np.array(data[metric_key]['values'])
            
            ax.plot(steps, values, label=f'Seed {seed}', 
                   color=color, alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Environment Steps', fontsize=11)
        ax.set_ylabel('Average Episode Reward', fontsize=11)
        ax.set_title(f'{algo_names[algo]} ({len(seed_data)} seeds)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved individual runs plot to: {output_path}")
    plt.close()


def plot_comparison(algo_data, metric_key, output_path):
    """Plot comparison of algorithms with available data."""
    plt.figure(figsize=(12, 7))
    
    colors = {'happo': '#1f77b4', 'mappo': '#ff7f0e', 'hatd3': '#2ca02c'}
    labels = {'happo': 'HAPPO', 'mappo': 'MAPPO', 'hatd3': 'HATD3'}
    
    for algo in ['happo', 'mappo', 'hatd3']:
        if algo not in algo_data:
            continue
        
        seed_data = algo_data[algo]
        all_curves = []
        all_steps = []
        
        for seed, data in seed_data.items():
            if metric_key in data:
                steps = np.array(data[metric_key]['steps'])
                values = np.array(data[metric_key]['values'])
                all_curves.append(values)
                all_steps.append(steps)
        
        if not all_curves:
            continue
        
        # Find common step range
        min_length = min(len(c) for c in all_curves)
        
        if min_length == 0:
            continue
        
        # Truncate all curves to same length
        curves = np.array([c[:min_length] for c in all_curves])
        steps = all_steps[0][:min_length]
        
        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)
        
        color = colors.get(algo, 'gray')
        label = f"{labels.get(algo, algo.upper())} ({len(all_curves)} seeds)"
        
        # Plot mean line
        plt.plot(steps, mean, label=label, color=color, linewidth=2)
        
        # Plot confidence interval (mean ± std)
        plt.fill_between(steps, mean - std, mean + std, 
                         color=color, alpha=0.2)
    
    plt.xlabel('Environment Steps', fontsize=12)
    plt.ylabel('Average Episode Reward', fontsize=12)
    plt.title('Partial Results - Algorithm Comparison (In Progress)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to: {output_path}")
    plt.close()


def generate_progress_report(algo_data, metric_key, output_path):
    """Generate a markdown report showing current progress."""
    lines = []
    
    lines.append("# Partial Experiment Results (In Progress)\n")
    lines.append(f"**Generated at**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    lines.append("## Progress Overview\n")
    lines.append("| Algorithm | Seeds Available | Latest Step | Current Reward |")
    lines.append("|-----------|-----------------|-------------|----------------|")
    
    for algo in ['happo', 'mappo', 'hatd3']:
        if algo not in algo_data:
            lines.append(f"| {algo.upper()} | 0 | - | - |")
            continue
        
        seed_data = algo_data[algo]
        num_seeds = len(seed_data)
        
        # Get latest step and reward
        latest_step = 0
        latest_reward = 0
        
        for seed, data in seed_data.items():
            if metric_key in data and len(data[metric_key]['steps']) > 0:
                step = data[metric_key]['steps'][-1]
                reward = data[metric_key]['values'][-1]
                if step > latest_step:
                    latest_step = step
                    latest_reward = reward
        
        lines.append(f"| {algo.upper()} | {num_seeds} | {latest_step:,} | {latest_reward:.2f} |")
    
    lines.append("\n## Detailed Progress\n")
    
    for algo in ['happo', 'mappo', 'hatd3']:
        if algo not in algo_data:
            continue
        
        lines.append(f"### {algo.upper()}\n")
        
        seed_data = algo_data[algo]
        
        for seed in sorted(seed_data.keys()):
            data = seed_data[seed]
            
            if metric_key in data and len(data[metric_key]['steps']) > 0:
                steps = data[metric_key]['steps']
                values = data[metric_key]['values']
                
                lines.append(f"**Seed {seed}**:")
                lines.append(f"- Current Step: {steps[-1]:,}")
                lines.append(f"- Current Reward: {values[-1]:.2f}")
                lines.append(f"- Data Points: {len(steps)}")
                lines.append(f"- Best Reward: {max(values):.2f}")
                lines.append("")
    
    lines.append("## Visualizations\n")
    lines.append("- `partial_comparison.png` - Algorithm comparison with available data")
    lines.append("- `partial_individual_runs.png` - Individual seed curves\n")
    
    lines.append("---\n")
    lines.append("*Note: This is a partial report. Final results will be available after all experiments complete.*\n")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Saved progress report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize partial experiment results during training"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Results directory (e.g., comparison_results/mamujoco_Humanoid-v2_17x1)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="eval/average_episode_rewards",
        help="Metric to plot (default: eval/average_episode_rewards)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <results_dir>/partial_analysis)"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "partial_analysis")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Visualizing Partial Experiment Results")
    print("="*60)
    print(f"Results Directory: {args.results_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Metric: {args.metric}")
    print("="*60 + "\n")
    
    # Aggregate available results
    print("Aggregating available results...")
    algo_data = aggregate_partial_results(args.results_dir)
    
    if not algo_data:
        print("\nERROR: No data found to visualize")
        print("\nTroubleshooting:")
        print("1. Check that experiments have started and are generating TensorBoard logs")
        print("2. Verify the results directory path is correct")
        print("3. Ensure TensorBoard logging is enabled in your configs")
        return
    
    # Show available metrics
    print("\nAvailable metrics:")
    metrics = get_available_metrics(algo_data)
    for i, metric in enumerate(metrics, 1):
        print(f"  {i}. {metric}")
    
    # Check if requested metric exists
    if args.metric not in metrics:
        print(f"\nWARNING: Metric '{args.metric}' not found in data.")
        print(f"Available metrics: {', '.join(metrics)}")
        
        # Try common alternatives
        alternatives = [m for m in metrics if 'reward' in m.lower() or 'return' in m.lower()]
        if alternatives:
            args.metric = alternatives[0]
            print(f"Using alternative metric: {args.metric}")
        else:
            print("No suitable metric found. Exiting.")
            return
    
    print(f"\nUsing metric: {args.metric}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Individual runs plot
    individual_plot_path = os.path.join(args.output_dir, "partial_individual_runs.png")
    plot_individual_runs(algo_data, args.metric, individual_plot_path)
    
    # Comparison plot
    comparison_plot_path = os.path.join(args.output_dir, "partial_comparison.png")
    plot_comparison(algo_data, args.metric, comparison_plot_path)
    
    # Progress report
    print("\nGenerating progress report...")
    report_path = os.path.join(args.output_dir, "progress_report.md")
    generate_progress_report(algo_data, args.metric, report_path)
    
    print("\n" + "="*60)
    print("Partial Analysis Complete!")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - {os.path.basename(comparison_plot_path)}")
    print(f"  - {os.path.basename(individual_plot_path)}")
    print(f"  - {os.path.basename(report_path)}")
    print("\nYou can run this script again as experiments progress to see updated results.")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Add pandas import that was missing
    try:
        import pandas as pd
    except ImportError:
        # Fallback if pandas not available
        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    from datetime import datetime
                    return datetime.now()
    
    main()
