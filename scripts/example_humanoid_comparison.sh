#!/bin/bash
# Example script for running comparison experiments on Humanoid-v2-17x1
# This demonstrates the complete workflow from config generation to analysis

set -e

echo "============================================="
echo "Humanoid-v2-17x1 Comparison Experiment"
echo "============================================="
echo ""

# Configuration
ENV="mamujoco"
SCENARIO="Humanoid-v2"
AGENT_CONF="17x1"
NUM_SEEDS=3
NUM_ENV_STEPS=10000000

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

# Step 1: Generate comparison configurations
echo "Step 1: Generating comparison configurations..."
echo "-------------------------------------------"
python scripts/generate_comparison_configs.py \
    --env "$ENV" \
    --scenario "$SCENARIO" \
    --agent_conf "$AGENT_CONF" \
    --num_env_steps "$NUM_ENV_STEPS" \
    --seeds 1 2 3

echo ""
echo "✓ Configurations generated successfully"
echo ""

# Step 2: Show what will be run
echo "Step 2: Experiment Setup"
echo "-------------------------------------------"
echo "The following experiments will be run:"
echo "  - HAPPO with seeds: 1, 2, 3"
echo "  - MAPPO with seeds: 1, 2, 3"
echo "  - HATD3 with seeds: 1, 2, 3"
echo ""
echo "Environment: Humanoid-v2-17x1 (MAMuJoCo)"
echo "Training steps: 10,000,000 per run"
echo "Total runs: 9"
echo ""
echo "⚠️  WARNING: Each run may take 24-48 hours on GPU!"
echo "⚠️  Make sure you have sufficient compute resources"
echo ""

# Ask for confirmation
read -p "Do you want to proceed with running experiments? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo ""
    echo "Experiment cancelled. Configs are saved in:"
    echo "  comparison_configs/mamujoco_Humanoid-v2_17x1/"
    echo ""
    echo "To run experiments later, use:"
    echo "  bash scripts/run_comparison_experiment.sh $ENV $SCENARIO $AGENT_CONF $NUM_SEEDS"
    exit 0
fi

echo ""
echo "Step 3: Running experiments..."
echo "-------------------------------------------"

# Step 3: Run experiments
bash scripts/run_comparison_experiment.sh "$ENV" "$SCENARIO" "$AGENT_CONF" "$NUM_SEEDS"

echo ""
echo "✓ All experiments completed"
echo ""

# Step 4: Analyze results
echo "Step 4: Analyzing results..."
echo "-------------------------------------------"

RESULTS_DIR="comparison_results/${ENV}_${SCENARIO}_${AGENT_CONF}"

python scripts/analyze_comparison_results.py \
    --exp_dir "$RESULTS_DIR"

echo ""
echo "✓ Analysis completed"
echo ""

# Step 5: Summary
echo "============================================="
echo "Experiment Complete!"
echo "============================================="
echo ""
echo "Results location:"
echo "  Config: comparison_configs/${ENV}_${SCENARIO}_${AGENT_CONF}/"
echo "  Results: ${RESULTS_DIR}/"
echo "  Analysis: ${RESULTS_DIR}/analysis/"
echo ""
echo "View the analysis:"
echo "  Report: ${RESULTS_DIR}/analysis/comparison_report.md"
echo "  Plots: ${RESULTS_DIR}/analysis/learning_curves.png"
echo ""
echo "============================================="
