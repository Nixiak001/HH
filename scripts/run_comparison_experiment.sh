#!/bin/bash
# Batch experiment runner for comparing HAPPO, MAPPO, and HATD3 algorithms
#
# Usage: bash run_comparison_experiment.sh <env> <scenario> <agent_conf> <num_seeds>
# Example: bash run_comparison_experiment.sh mamujoco Humanoid-v2 17x1 3

set -e

# Get arguments
ENV=${1:-mamujoco}
SCENARIO=${2:-Humanoid-v2}
AGENT_CONF=${3:-17x1}
NUM_SEEDS=${4:-3}

# Get script directory and repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Set paths
CONFIG_DIR="${REPO_ROOT}/comparison_configs/${ENV}_${SCENARIO}_${AGENT_CONF}"
RESULTS_DIR="${REPO_ROOT}/comparison_results/${ENV}_${SCENARIO}_${AGENT_CONF}"
EXAMPLES_DIR="${REPO_ROOT}/examples"

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory not found: $CONFIG_DIR"
    echo "Please run generate_comparison_configs.py first:"
    echo "  python scripts/generate_comparison_configs.py --env $ENV --scenario $SCENARIO --agent_conf $AGENT_CONF"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Read metadata
METADATA_FILE="${CONFIG_DIR}/experiment_metadata.json"
if [ ! -f "$METADATA_FILE" ]; then
    echo "Error: Metadata file not found: $METADATA_FILE"
    exit 1
fi

# Extract seeds from metadata using python
SEEDS=$(python3 -c "import json; f=open('${METADATA_FILE}'); d=json.load(f); print(' '.join(map(str, d['seeds'][:${NUM_SEEDS}])))")

echo "============================================="
echo "Starting Comparison Experiment"
echo "============================================="
echo "Environment: $ENV"
echo "Scenario: $SCENARIO"
echo "Agent Config: $AGENT_CONF"
echo "Seeds: $SEEDS"
echo "Config Dir: $CONFIG_DIR"
echo "Results Dir: $RESULTS_DIR"
echo "============================================="
echo ""

# Algorithms to run
ALGORITHMS=("happo" "mappo" "hatd3")

# Run experiments for each algorithm and seed
for ALGO in "${ALGORITHMS[@]}"; do
    CONFIG_FILE="${CONFIG_DIR}/${ALGO}_comparison.json"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "WARNING: Config file not found for $ALGO, skipping: $CONFIG_FILE"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Running $ALGO"
    echo "=========================================="
    
    for SEED in $SEEDS; do
        EXP_NAME="comparison_${ALGO}_seed${SEED}"
        
        echo ""
        echo "----------------------------------------"
        echo "Algorithm: $ALGO | Seed: $SEED"
        echo "Experiment: $EXP_NAME"
        echo "----------------------------------------"
        
        # Change to examples directory
        cd "$EXAMPLES_DIR"
        
        # Run training
        python train.py \
            --load_config "$CONFIG_FILE" \
            --exp_name "$EXP_NAME" \
            --seed "$SEED" \
            2>&1 | tee "${RESULTS_DIR}/${ALGO}_seed${SEED}_log.txt"
        
        echo "✓ Completed: $ALGO with seed $SEED"
    done
    
    echo ""
    echo "✓ Completed all runs for $ALGO"
done

echo ""
echo "============================================="
echo "All Experiments Completed!"
echo "============================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Next step: Analyze results using:"
echo "  python scripts/analyze_comparison_results.py --exp_dir $RESULTS_DIR"
echo "============================================="
