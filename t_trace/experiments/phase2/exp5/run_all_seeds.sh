#!/bin/bash
# Run Phase 2 Experiment 5 across 5 random seeds with sequential execution
# Leverages Ryzen 9 7900X (12 cores) for efficient processing

set -e

echo "=============================================="
echo "M-TRACE Phase 2 Experiment 5: Statistical Rigor"
echo "5 Random Seeds with Sequential Execution"
echo "=============================================="

SEEDS=(42 123 456 789 1011)
RESULTS_DIR="t_trace/experiments/phase2/exp5/results/raw"

# Create results directory
mkdir -p $RESULTS_DIR

# Function to run single seed
run_seed() {
    local seed=$1
    echo "🔄 Starting seed $seed..."
    python t_trace/experiments/phase2/exp5/run_experiment_single_seed.py \
        --seed $seed \
        --dataset breast_cancer \
        --model-type mlp
    echo "✅ Seed $seed complete"
}

export -f run_seed

# Sequential execution (CPU-based, no GPU memory concerns)
echo "Running seeds sequentially..."
for seed in "${SEEDS[@]}"; do
    run_seed $seed
done

# Aggregate results
echo ""
echo "📊 Aggregating results across all seeds..."
python t_trace/experiments/phase2/exp5/statistical_analysis.py

echo ""
echo "=============================================="
echo "✅ Statistical Rigor Complete!"
echo "Results saved to: t_trace/experiments/phase2/exp5/results/"
echo "=============================================="