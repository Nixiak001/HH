#!/usr/bin/env python
"""
Analyze and visualize comparison experiment results.

This script:
- Parses TensorBoard logs from multiple algorithm runs
- Computes statistical metrics (mean, std, final performance)
- Generates comparative learning curves with confidence intervals
- Creates a markdown report with tables and plots
"""

import argparse
import os
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("WARNING: tensorboard not available. Install with: pip install tensorboard")


def find_tensorboard_logs(results_dir):
    """Find all TensorBoard event files in the results directory."""
    pattern = os.path.join(results_dir, "**", "events.out.tfevents.*")
    event_files = glob.glob(pattern, recursive=True)
    return event_files


def parse_tensorboard_log(event_file):
    """Parse a TensorBoard event file and extract scalar metrics."""
    if not HAS_TENSORBOARD:
        return {}
    
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


def extract_algo_seed_from_path(path):
    """Extract algorithm name and seed from result path."""
    parts = path.split('/')
    
    # Look for directory or file with algo and seed info
    for part in reversed(parts):
        if 'happo' in part.lower():
            algo = 'happo'
        elif 'mappo' in part.lower():
            algo = 'mappo'
        elif 'hatd3' in part.lower():
            algo = 'hatd3'
        else:
            continue
        
        # Extract seed
        if 'seed' in part.lower():
            try:
                seed = int(part.split('seed')[-1].split('_')[0])
                return algo, seed
            except:
                pass
    
    return None, None


def aggregate_results(results_dir):
    """Aggregate results from all algorithm runs."""
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
            print(f"WARNING: Could not extract algo/seed from: {event_file}")
            continue
        
        print(f"  Processing: {algo} seed {seed}")
        data = parse_tensorboard_log(event_file)
        
        if data:
            algo_data[algo][seed] = data
    
    return dict(algo_data)


def compute_statistics(algo_data, metric_key='eval/average_episode_rewards'):
    """Compute mean and std across seeds for each algorithm."""
    stats = {}
    
    for algo, seed_data in algo_data.items():
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
        
        # Truncate all curves to same length
        curves = np.array([c[:min_length] for c in all_curves])
        steps = all_steps[0][:min_length]
        
        stats[algo] = {
            'steps': steps,
            'mean': np.mean(curves, axis=0),
            'std': np.std(curves, axis=0),
            'min': np.min(curves, axis=0),
            'max': np.max(curves, axis=0),
            'final_mean': np.mean(curves[:, -1]),
            'final_std': np.std(curves[:, -1]),
            'num_seeds': len(all_curves)
        }
    
    return stats


def plot_comparison(stats, output_path, title="Algorithm Comparison"):
    """Plot learning curves with confidence intervals."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not available)")
        return
    
    plt.figure(figsize=(12, 7))
    
    colors = {'happo': '#1f77b4', 'mappo': '#ff7f0e', 'hatd3': '#2ca02c'}
    labels = {'happo': 'HAPPO', 'mappo': 'MAPPO', 'hatd3': 'HATD3'}
    
    for algo in ['happo', 'mappo', 'hatd3']:
        if algo not in stats:
            continue
        
        s = stats[algo]
        steps = s['steps']
        mean = s['mean']
        std = s['std']
        
        color = colors.get(algo, 'gray')
        label = labels.get(algo, algo.upper())
        
        # Plot mean line
        plt.plot(steps, mean, label=label, color=color, linewidth=2)
        
        # Plot confidence interval (mean ± std)
        plt.fill_between(steps, mean - std, mean + std, 
                         color=color, alpha=0.2)
    
    plt.xlabel('Environment Steps', fontsize=12)
    plt.ylabel('Average Episode Reward', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def generate_report(stats, output_path, exp_info=None):
    """Generate a markdown report with tables and statistics."""
    lines = []
    
    lines.append("# Comparison Experiment Results\n")
    
    if exp_info:
        lines.append("## Experiment Configuration\n")
        lines.append(f"- **Environment**: {exp_info.get('env', 'N/A')}")
        lines.append(f"- **Scenario**: {exp_info.get('scenario', 'N/A')}")
        lines.append(f"- **Agent Config**: {exp_info.get('agent_conf', 'N/A')}")
        lines.append(f"- **Training Steps**: {exp_info.get('num_env_steps', 'N/A')}")
        lines.append(f"- **Seeds**: {exp_info.get('seeds', 'N/A')}\n")
    
    lines.append("## Performance Summary\n")
    lines.append("| Algorithm | Final Mean Reward | Std Dev | Num Seeds |")
    lines.append("|-----------|-------------------|---------|-----------|")
    
    # Sort by final mean reward
    sorted_algos = sorted(stats.items(), 
                         key=lambda x: x[1]['final_mean'], 
                         reverse=True)
    
    for algo, s in sorted_algos:
        lines.append(f"| {algo.upper()} | {s['final_mean']:.2f} | "
                    f"{s['final_std']:.2f} | {s['num_seeds']} |")
    
    lines.append("\n## Detailed Statistics\n")
    
    for algo, s in sorted_algos:
        lines.append(f"### {algo.upper()}\n")
        lines.append(f"- **Final Performance**: {s['final_mean']:.2f} ± {s['final_std']:.2f}")
        lines.append(f"- **Number of Seeds**: {s['num_seeds']}")
        lines.append(f"- **Total Steps**: {s['steps'][-1]}")
        lines.append(f"- **Max Reward**: {np.max(s['max']):.2f}")
        lines.append(f"- **Min Reward**: {np.min(s['min']):.2f}\n")
    
    lines.append("## Learning Curves\n")
    lines.append("See `learning_curves.png` for comparative visualization.\n")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Saved report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze comparison experiment results"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment results directory"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="eval/average_episode_rewards",
        help="Metric key to analyze (default: eval/average_episode_rewards)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: <exp_dir>/analysis)"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.exp_dir, "analysis")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("Analyzing Comparison Experiment Results")
    print("="*50)
    print(f"Results Directory: {args.exp_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Metric: {args.metric}")
    print("="*50 + "\n")
    
    # Aggregate results
    print("Aggregating results...")
    algo_data = aggregate_results(args.exp_dir)
    
    if not algo_data:
        print("ERROR: No data found to analyze")
        return
    
    print(f"\nFound data for algorithms: {list(algo_data.keys())}")
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(algo_data, args.metric)
    
    if not stats:
        print("ERROR: Could not compute statistics")
        return
    
    # Try to load experiment metadata
    exp_info = None
    metadata_path = os.path.join(
        os.path.dirname(args.exp_dir), 
        "experiment_metadata.json"
    )
    if not os.path.exists(metadata_path):
        # Try looking in parent of parent
        parts = args.exp_dir.rstrip('/').split('/')
        if 'comparison_results' in parts:
            idx = parts.index('comparison_results')
            exp_name = parts[idx + 1] if idx + 1 < len(parts) else None
            if exp_name:
                metadata_path = os.path.join(
                    '/'.join(parts[:idx]),
                    'comparison_configs',
                    exp_name,
                    'experiment_metadata.json'
                )
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            exp_info = json.load(f)
    
    # Generate plot
    print("\nGenerating plots...")
    plot_path = os.path.join(args.output_dir, "learning_curves.png")
    title = "Algorithm Comparison"
    if exp_info:
        title = f"{exp_info.get('scenario', '')} {exp_info.get('agent_conf', '')} Comparison"
    plot_comparison(stats, plot_path, title)
    
    # Generate report
    print("\nGenerating report...")
    report_path = os.path.join(args.output_dir, "comparison_report.md")
    generate_report(stats, report_path, exp_info)
    
    # Save statistics as JSON
    stats_json = {}
    for algo, s in stats.items():
        stats_json[algo] = {
            'final_mean': float(s['final_mean']),
            'final_std': float(s['final_std']),
            'num_seeds': int(s['num_seeds']),
            'total_steps': int(s['steps'][-1])
        }
    
    stats_path = os.path.join(args.output_dir, "statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_json, f, indent=2)
    print(f"✓ Saved statistics to: {stats_path}")
    
    print("\n" + "="*50)
    print("Analysis Complete!")
    print("="*50)
    print(f"Results saved to: {args.output_dir}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
