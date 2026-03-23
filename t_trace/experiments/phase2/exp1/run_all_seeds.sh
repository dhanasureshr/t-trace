#!/bin/bash
# Run Phase 2 Experiment 1 across 5 random seeds with parallel execution
# Leverages Ryzen 9 7900X (12 cores) for efficient processing

set -e

echo "=============================================="
echo "M-TRACE Phase 2 Experiment 1: Statistical Rigor"
echo "5 Random Seeds with Parallel Execution"
echo "=============================================="

SEEDS=(42 123 456 789 1011)
RESULTS_DIR="t_trace/experiments/phase2/exp1/results/raw"

# Create results directory
mkdir -p $RESULTS_DIR

# Function to run single seed
run_seed() {
    local seed=$1
    echo "🔄 Starting seed $seed..."
    python t_trace/experiments/phase2/exp1/validation_protocol_single_seed.py --seed $seed
    echo "✅ Seed $seed complete"
}

export -f run_seed

# Option 1: Sequential (safer for GPU memory)
echo "Running seeds sequentially (GPU memory safe)..."
for seed in "${SEEDS[@]}"; do
    run_seed $seed
done

# Option 2: Parallel (faster, requires more GPU memory)
# Uncomment below if you have sufficient GPU memory
# echo "Running seeds in parallel (requires 16GB+ GPU)..."
# parallel -j 4 run_seed ::: "${SEEDS[@]}"

# Aggregate results
echo ""
echo "📊 Aggregating results across all seeds..."
python t_trace/experiments/phase2/exp1/statistical_analysis.py

echo ""
echo "=============================================="
echo "✅ Statistical Rigor Complete!"
echo "Results saved to: t_trace/experiments/phase2/exp1/results/"
echo "=============================================="