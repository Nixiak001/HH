# Comparison Experiment Framework

This framework provides automated tools for comparing HAPPO, MAPPO, and HATD3 algorithms under identical environment and reward configurations.

## Overview

The framework consists of three main components:

1. **Configuration Generator** (`generate_comparison_configs.py`) - Creates unified configs for fair comparison
2. **Batch Experiment Runner** (`run_comparison_experiment.sh`) - Executes training across multiple seeds
3. **Results Analysis Tool** (`analyze_comparison_results.py`) - Generates comparative visualizations and reports

## Quick Start for Humanoid-v2-17x1

### Step 1: Generate Comparison Configurations

```bash
cd /home/runner/work/HH/HH
python scripts/generate_comparison_configs.py \
    --env mamujoco \
    --scenario Humanoid-v2 \
    --agent_conf 17x1 \
    --num_env_steps 10000000 \
    --seeds 1 2 3
```

This creates unified configurations in `comparison_configs/mamujoco_Humanoid-v2_17x1/` that ensure:
- Same environment and scenario (Humanoid-v2 with 17x1 agent configuration)
- Same total training steps (10M by default)
- Consistent evaluation intervals and seeds
- Algorithm-specific parameters preserved from tuned configs

### Step 2: Run Comparison Experiments

```bash
bash scripts/run_comparison_experiment.sh mamujoco Humanoid-v2 17x1 3
```

This will:
- Run HAPPO, MAPPO, and HATD3 with 3 random seeds each (9 total runs)
- Save results to `comparison_results/mamujoco_Humanoid-v2_17x1/`
- Generate training logs for each run

**Note**: Each run may take several hours depending on your hardware. The Humanoid-v2-17x1 environment is particularly compute-intensive.

### Step 3: Analyze Results

```bash
python scripts/analyze_comparison_results.py \
    --exp_dir comparison_results/mamujoco_Humanoid-v2_17x1
```

This generates:
- `analysis/learning_curves.png` - Comparative learning curves with confidence intervals
- `analysis/comparison_report.md` - Detailed performance statistics
- `analysis/statistics.json` - Raw statistics in JSON format

## Using Other Environments

The framework supports any environment in the repository. Examples:

### Ant-v2-4x2
```bash
# Generate configs
python scripts/generate_comparison_configs.py \
    --env mamujoco --scenario Ant-v2 --agent_conf 4x2

# Run experiments
bash scripts/run_comparison_experiment.sh mamujoco Ant-v2 4x2 3

# Analyze
python scripts/analyze_comparison_results.py \
    --exp_dir comparison_results/mamujoco_Ant-v2_4x2
```

### HalfCheetah-v2-6x1
```bash
# Generate configs
python scripts/generate_comparison_configs.py \
    --env mamujoco --scenario HalfCheetah-v2 --agent_conf 6x1

# Run experiments
bash scripts/run_comparison_experiment.sh mamujoco HalfCheetah-v2 6x1 3

# Analyze
python scripts/analyze_comparison_results.py \
    --exp_dir comparison_results/mamujoco_HalfCheetah-v2_6x1
```

## Advanced Options

### Custom Training Steps

```bash
python scripts/generate_comparison_configs.py \
    --env mamujoco \
    --scenario Humanoid-v2 \
    --agent_conf 17x1 \
    --num_env_steps 50000000  # 50M steps
```

### More Random Seeds

```bash
python scripts/generate_comparison_configs.py \
    --env mamujoco \
    --scenario Humanoid-v2 \
    --agent_conf 17x1 \
    --seeds 1 2 3 4 5  # 5 seeds
```

### Custom Experiment Name

```bash
python scripts/generate_comparison_configs.py \
    --env mamujoco \
    --scenario Humanoid-v2 \
    --agent_conf 17x1 \
    --exp_name my_experiment
```

### Analyzing Specific Metrics

```bash
python scripts/analyze_comparison_results.py \
    --exp_dir comparison_results/mamujoco_Humanoid-v2_17x1 \
    --metric train/average_episode_rewards
```

## Directory Structure

```
/home/runner/work/HH/HH/
├── scripts/
│   ├── generate_comparison_configs.py    # Config generator
│   ├── run_comparison_experiment.sh      # Batch runner
│   └── analyze_comparison_results.py     # Results analyzer
├── comparison_configs/
│   └── mamujoco_Humanoid-v2_17x1/
│       ├── happo_comparison.json
│       ├── mappo_comparison.json
│       ├── hatd3_comparison.json
│       └── experiment_metadata.json
└── comparison_results/
    └── mamujoco_Humanoid-v2_17x1/
        ├── happo/
        ├── mappo/
        ├── hatd3/
        └── analysis/
            ├── learning_curves.png
            ├── comparison_report.md
            └── statistics.json
```

## Requirements

The analysis script requires additional Python packages:

```bash
pip install matplotlib seaborn tensorboard pandas
```

If these are not installed, the experiment can still run, but visualization will be skipped.

## Troubleshooting

### No TensorBoard logs found
- Check that experiments completed successfully
- Verify the results directory path
- Ensure TensorBoard logging is enabled in configs

### Config not found error
- Run `generate_comparison_configs.py` before running experiments
- Check that the environment/scenario/agent_conf match between config generation and experiment execution

### Out of memory during training
- Reduce `n_rollout_threads` in configs
- Use fewer parallel environments
- Consider running algorithms sequentially instead of in parallel

## Performance Notes

### Humanoid-v2-17x1 Specific
- Training is compute-intensive (expect 24-48 hours per seed on GPU)
- Requires significant memory (8GB+ GPU recommended)
- HAPPO config uses recurrent policy which adds overhead
- HATD3 is an off-policy algorithm and may converge faster

### General Tips
- Use CUDA if available (set `cuda: true` in configs)
- Monitor GPU memory usage during training
- Use `screen` or `tmux` for long-running experiments
- Save checkpoints regularly in case of interruption

## Citation

If you use this framework for research, please cite the HARL paper:

```bibtex
@article{JMLR:v25:23-0488,
  author  = {Yifan Zhong and Jakub Grudzien Kuba and Xidong Feng and Siyi Hu and Jiaming Ji and Yaodong Yang},
  title   = {Heterogeneous-Agent Reinforcement Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {32},
  pages   = {1--67},
  url     = {http://jmlr.org/papers/v25/23-0488.html}
}
```
