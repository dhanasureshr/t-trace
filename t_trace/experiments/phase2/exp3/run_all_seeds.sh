#!/bin/bash
# Run Phase 2 Experiment 3 across 5 random seeds with parallel execution
# Leverages Ryzen 9 7900X (12 cores) for efficient processing

set -e

echo "=============================================="
echo "M-TRACE Phase 2 Experiment 3: Statistical Rigor"
echo "5 Random Seeds with Sequential Execution"
echo "=============================================="

SEEDS=(42 123 456 789 1011)
RESULTS_DIR="t_trace/experiments/phase2/exp3/results/raw"

# Create results directory
mkdir -p $RESULTS_DIR

# Function to run single seed
run_seed() {
    local seed=$1
    echo "🔄 Starting seed $seed..."
    python t_trace/experiments/phase2/exp3/run_experiment_single_seed.py --seed $seed
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
python t_trace/experiments/phase2/exp3/statistical_analysis.py

echo ""
echo "=============================================="
echo "✅ Statistical Rigor Complete!"
echo "Results saved to: t_trace/experiments/phase2/exp3/results/"
echo "=============================================="