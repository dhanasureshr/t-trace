#!/bin/bash
# Run Phase 2 Experiment 2 across 5 random seeds with parallel execution
# Leverages Ryzen 9 7900X (12 cores) for efficient processing

set -e

echo "=============================================="
echo "M-TRACE Phase 2 Experiment 2: Statistical Rigor"
echo "5 Random Seeds with Sequential Execution"
echo "=============================================="

SEEDS=(42 123 456 789 1011)
RESULTS_DIR="t_trace/experiments/phase2/exp2/results/raw"

# Create results directory
mkdir -p $RESULTS_DIR

# Function to run single seed
run_seed() {
    local seed=$1
    echo "🔄 Starting seed $seed..."
    python t_trace/experiments/phase2/exp2/run_experiment_single_seed.py --seed $seed
    echo "✅ Seed $seed complete"
    
    # Clear GPU cache to prevent OOM on next seed
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

export -f run_seed

# Sequential execution (GPU memory safe for BERT training)
echo "Running seeds sequentially (GPU memory safe for BERT training)..."
for seed in "${SEEDS[@]}"; do
    run_seed $seed
done

# Aggregate results
echo ""
echo "📊 Aggregating results across all seeds..."
python t_trace/experiments/phase2/exp2/statistical_analysis.py

echo ""
echo "=============================================="
echo "✅ Statistical Rigor Complete!"
echo "Results saved to: t_trace/experiments/phase2/exp2/results/"
echo "=============================================="