# M-TRACE Artifact Package (Submission Version)

> **Model Transparency through Recursive Analysis of Contextual Encapsulation**  
> **Submission:** FAccT 2027 / NeurIPS Datasets & Benchmarks  
> **Version:** 1.0.0 (Anonymous Submission)  
> **License:** MIT (Anonymized)

## 🎯 Core Scientific Claim
This artifact validates the claim: *"M-TRACE captures the model's actual computational trajectory during inference—revealing when/where reasoning occurs—while post-hoc tools fundamentally cannot access this temporal dimension."*

## ⚡ Quick Start (Evaluation Workstation)

### Prerequisites
- Ubuntu 22.04+
- NVIDIA Driver 535+ (CUDA 12.3)
- Docker + NVIDIA Container Toolkit

### Execution Steps
1. **Build Container:**
   ```bash
   cd artifact
   docker build -t mtrace-artifact:latest -f docker/Dockerfile .