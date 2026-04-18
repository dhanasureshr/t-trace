#!/bin/bash
# run_all_experiments.sh
set -e

echo "=== M-TRACE Reproducibility Package ==="

# Scientific Validation (The core contribution)
echo "Running Experiment 1: Measuring Temporal Grounding"
python t_trace/experiments/phase2/Experiment1/exp1_temporal_grounding_scaled.py --all # This will run all seeds + aggregation for Experiment 1

echo "Running Experiment 2: Measuring Attribution-Causality Divergence"
python t_trace/experiments/phase2/Experiment2/exp2_causal_verification.py --all # This will run all seeds + aggregation for Experiment 2

echo "Running Experiment 3: Identifying Boundary conditions"
python t_trace/experiments/phase2/Experiment3/exp3_boundary_condition_analysis.py --all # This will run all seeds + aggregation for Experiment 3


echo "=== All Experiments Complete ==="
echo "Results aggregated in: t_trace/experiments/phase2/Experiment1/results/"
echo "Results aggregated in: t_trace/experiments/phase2/Experiment2/results/"
echo "Results aggregated in: t_trace/experiments/phase2/Experiment3/results/"