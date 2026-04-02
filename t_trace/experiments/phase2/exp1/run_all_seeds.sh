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


echo "=============================================="
echo "PHASE 2 EXP 1: ROBUSTNESS VALIDATION (--noise-sigma)"
echo "=============================================="

# Define noise run variables
NOISE_SIGMA="0.15"
RESULTS_DIR_NOISE="t_trace/experiments/phase2/exp1/results/raw_noise"
mkdir -p "$RESULTS_DIR_NOISE"

run_seed_noisy() {
    local seed=$1
    echo "Starting NOISY seed 🔄 $seed..."
    # Pass the noise argument
    python t_trace/experiments/phase2/exp1/validation_protocol_single_seed.py \
        --seed $seed \
        --results-dir "$RESULTS_DIR_NOISE" \
        --noise-sigma $NOISE_SIGMA || true
    echo "Seed ✅ $seed complete"
}

export -f run_seed_noisy
export RESULTS_DIR_NOISE
export NOISE_SIGMA

# Sequential execution (CPU/GPU safe)
for seed in "${SEEDS[@]}"; do run_seed_noisy $seed; done

echo "Aggregating NOISY results..."
# You may need to adapt the aggregation command to handle the different directory
# python t_trace/experiments/phase2/exp1/statistical_analysis.py --data-source-noise