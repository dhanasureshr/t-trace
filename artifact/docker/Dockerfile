cat > docker/Dockerfile << 'EOF'
# =============================================================================
# M-TRACE Artifact Reproducibility Package
# Hardware: Ubuntu + RTX 4080 Super + Ryzen 9 7900X + 64GB DDR5
# CUDA: 12.1 | cuDNN: 9.1 | Python: 3.10+
# Validated: 2026-03-24 (Phase 1 Complete)
# =============================================================================

FROM nvidia/cuda:12.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Optimized for RTX 4080 Super (Compute Capability 8.9)
ENV TORCH_CUDA_ARCH_LIST="8.9"

# Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git curl wget libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    build-essential cmake \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app/artifact

# Copy requirements first (layer caching optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copy artifact code (parent directory contains t_trace package)
COPY .. /app/artifact

# Install M-TRACE package in editable mode
RUN pip3 install -e .

# Pre-warm CUDA kernels for RTX 4080 Super
RUN python3 -c "import torch; torch.ones(1).cuda(); print('CUDA Warmup Complete')"

# Environment variables for reproducibility
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=12
ENV MKL_NUM_THREADS=12
ENV NUMEXPR_NUM_THREADS=12

# Health check for container readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available(); print('GPU ready')"

# Default entrypoint
ENTRYPOINT ["python3", "-u"]
CMD ["scripts/validate_environment.py"]
EOF

# Verify Dockerfile was created
ls -la docker/Dockerfile