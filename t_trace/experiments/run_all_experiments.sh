#!/bin/bash
# run_all_experiments.sh
set -e

echo "=== M-TRACE Reproducibility Package ==="

# Phase 1: Core Validation
echo "Running Phase 1: PyTorch Validation..."
python t_trace/experiments/phase1/pytorch_validation/hook_validation.py

echo "Running Phase 1: Sklearn Validation..."
python t_trace/experiments/phase1/sklearn_validation/hook_validation.py

echo "Running Phase 1: tensorflow Validation..."
python t_trace/experiments/phase1/tensorflow_validation/hook_validation.py

# Phase 2: Scientific Validation (The core contribution)
echo "Running Phase 2, Exp 1: Temporal Fidelity..."
bash t_trace/experiments/phase2/exp1/run_all_seeds.sh

echo "Running Phase 2, Exp 2: Gradient-Attention Causality..."
bash t_trace/experiments/phase2/exp2/run_all_seeds.sh

echo "Running Phase 2, Exp 3: Decision Path Fidelity in Tree Ensembles.."
bash t_trace/experiments/phase2/exp3/run_all_seeds.sh

echo "Running Phase 2, Exp 4: Cross-modal reasoning transparency..."
bash t_trace/experiments/phase2/exp4/run_all_seeds.sh

echo "Running Phase 2, Exp 5: Tests the boundary condition of M-TRACE..."

python t_trace/experiments/phase2/exp5/exp5_static_aggregation_v2.py

# ... Repeat for Exp 3, 4, 5 ...

echo "=== All Experiments Complete ==="
echo "Results aggregated in: t_trace/experiments/phase2/exp1/results/"
echo "Results aggregated in: t_trace/experiments/phase2/exp2/results/"
echo "Results aggregated in: t_trace/experiments/phase2/exp3/results/"
echo "Results aggregated in: t_trace/experiments/phase2/exp4/results/"
echo "Results aggregated in: t_trace/experiments/phase2/exp5/"