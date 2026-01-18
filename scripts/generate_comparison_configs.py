#!/usr/bin/env python
"""
Generate unified comparison configurations for HAPPO, MAPPO, and HATD3 algorithms.

This script creates configuration files that ensure fair comparison by:
- Using the same environment and scenario
- Normalizing training steps and evaluation intervals
- Using consistent random seeds
- Preserving algorithm-specific parameters
"""

import argparse
import json
import os
import copy
from pathlib import Path


def load_tuned_config(base_path, env, scenario, agent_conf, algo):
    """Load existing tuned configuration for an algorithm."""
    config_path = os.path.join(
        base_path, 
        "tuned_configs", 
        env, 
        f"{scenario}-{agent_conf}", 
        algo, 
        "config.json"
    )
    
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)


def create_unified_config(base_config, algo, env, scenario, agent_conf, 
                          num_env_steps, seeds, exp_name_prefix):
    """Create a unified configuration with normalized parameters."""
    config = copy.deepcopy(base_config)
    
    # Update main args
    config["main_args"]["algo"] = algo
    config["main_args"]["env"] = env
    config["main_args"]["exp_name"] = f"{exp_name_prefix}_{algo}"
    config["main_args"]["load_config"] = ""
    
    # Update environment args
    config["env_args"]["scenario"] = scenario
    config["env_args"]["agent_conf"] = agent_conf
    
    # Update training steps
    if "train" in config["algo_args"]:
        config["algo_args"]["train"]["num_env_steps"] = num_env_steps
    
    # Update seed (will be overridden at runtime for multiple seeds)
    if "seed" in config["algo_args"]:
        config["algo_args"]["seed"]["seed"] = seeds[0]
        config["algo_args"]["seed"]["seed_specify"] = True
    
    # Ensure consistent output directory
    if "logger" in config["algo_args"]:
        config["algo_args"]["logger"]["log_dir"] = "./comparison_results"
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison configurations for HAPPO, MAPPO, and HATD3"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        default="mamujoco",
        help="Environment name (default: mamujoco)"
    )
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="Humanoid-v2",
        help="Scenario name (default: Humanoid-v2)"
    )
    parser.add_argument(
        "--agent_conf", 
        type=str, 
        default="17x1",
        help="Agent configuration (default: 17x1)"
    )
    parser.add_argument(
        "--num_env_steps", 
        type=int, 
        default=10000000,
        help="Total number of training steps (default: 10000000)"
    )
    parser.add_argument(
        "--seeds", 
        type=int, 
        nargs="+", 
        default=[1, 2, 3],
        help="Random seeds to use (default: 1 2 3)"
    )
    parser.add_argument(
        "--exp_name", 
        type=str, 
        default="comparison",
        help="Experiment name prefix (default: comparison)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="comparison_configs",
        help="Output directory for configs (default: comparison_configs)"
    )
    
    args = parser.parse_args()
    
    # Get repository root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    # Algorithms to compare
    algorithms = ["happo", "mappo", "hatd3"]
    
    # Create output directory
    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment-specific subdirectory
    exp_dir = os.path.join(
        output_dir, 
        f"{args.env}_{args.scenario}_{args.agent_conf}"
    )
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\nGenerating comparison configs for:")
    print(f"  Environment: {args.env}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Agent Config: {args.agent_conf}")
    print(f"  Training Steps: {args.num_env_steps}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Output Directory: {exp_dir}\n")
    
    # Generate configs for each algorithm
    configs_created = []
    for algo in algorithms:
        print(f"Processing {algo.upper()}...")
        
        # Load the tuned config
        base_config = load_tuned_config(
            repo_root, args.env, args.scenario, args.agent_conf, algo
        )
        
        if base_config is None:
            print(f"  WARNING: No tuned config found for {algo}, skipping.")
            continue
        
        # Create unified config
        unified_config = create_unified_config(
            base_config, 
            algo, 
            args.env, 
            args.scenario, 
            args.agent_conf,
            args.num_env_steps,
            args.seeds,
            args.exp_name
        )
        
        # Save config
        output_path = os.path.join(exp_dir, f"{algo}_comparison.json")
        with open(output_path, 'w') as f:
            json.dump(unified_config, f, indent=2)
        
        configs_created.append(output_path)
        print(f"  ✓ Created: {output_path}")
    
    # Create a metadata file
    metadata = {
        "env": args.env,
        "scenario": args.scenario,
        "agent_conf": args.agent_conf,
        "num_env_steps": args.num_env_steps,
        "seeds": args.seeds,
        "exp_name": args.exp_name,
        "algorithms": algorithms,
        "configs": configs_created
    }
    
    metadata_path = os.path.join(exp_dir, "experiment_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ All configs generated successfully!")
    print(f"✓ Metadata saved to: {metadata_path}")
    print(f"\nNext step: Run the experiments using:")
    print(f"  bash scripts/run_comparison_experiment.sh {args.env} {args.scenario} {args.agent_conf} {len(args.seeds)}")


if __name__ == "__main__":
    main()
