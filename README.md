# M-TRACE: Model Transparency through Recursive Analysis of Contextual Encapsulation (Anonymous Submission)


## 🚀 Reproducibility Instructions
All experiments are containerized. No local Python environment setup is required.
**Estimated Total Runtime:** ~2-3 hours (GPU), ~6-8 hours (CPU)



### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (Recommended for Phase 2. CPU fallback available for Phase 1)
- 16GB+ RAM (32GB recommended for Phase 2 Exp 4)

### Step 1: Build the Container
```bash
cd docker
docker compose up -d --build

```

### Step 2: Run All Experiments
# This script runs Phase 1 & Phase 2 experiments sequentially
```bash
docker compose exec mtrace-experiment bash t_trace/experiments/run_all_experiments.sh
```

### Step 3: View Results
Results are automatically saved to the `t_trace/experiments/.../results/` directories on your host machine.





