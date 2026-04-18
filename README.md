# M-TRACE: Model Transparency through Recursive Analysis of Contextual Encapsulation

## 🚀 Reproducibility Instructions
All experiments are containerized. No local Python environment setup is required.
**Estimated Total Runtime:** ~2-3 hours (GPU), ~6-8 hours (CPU)



### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (Recommended for reproducablity)
- 16GB+ RAM (32GB recommended)

### Step 1: Build the Container
```bash
cd docker
docker compose up -d --build
```

### Step 2: Run All Experiments
# This script runs Experiments sequentially
```bash
docker compose exec mtrace-experiment bash t_trace/experiments/run_all_experiments.sh
```

### Step 3: View Results
Results are automatically saved to the `t_trace/experiments/.../results/` directories on your host machine.

### Appendix A: Multi-Layer Patching Ablation script 
```bash
docker compose exec mtrace-experiment python t_trace/experiments/phase2/Experiment2/exp2_multi_layer_patching.py --all
```

### Appendix B: Attribution-Trajectory Correlation Noise Ablation script 
```bash
docker compose exec mtrace-experiment python t_trace/experiments/phase2/Experiment1/atc_noise_ablation.py
```

### Appendix C: Threshold Sensitivity Analysis script
```bash
docker compose exec mtrace-experiment python t_trace/experiments/phase2/Experiment2/exp2_threshold_sensitivity.py
```
