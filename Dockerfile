# =========================================================
# M-TRACE Docker Image (v5.2 - FINAL)
# Aligned with: Experimental Plan v3, Implementation v4
# Hardware: RTX 4080 Super (CUDA 12.3), Ryzen 9 7900X
# Python: 3.12 (via deadsnakes PPA on Ubuntu 22.04)
# =========================================================

FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Prevent Interactive Prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# === STEP 1: Install Python 3.12 via deadsnakes PPA ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    gnupg2 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        git \
        libsnappy-dev \
        libzstd-dev \
        libssl-dev \
        libffi-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# === STEP 2: Set Python 3.12 as default (ONCE) ===
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# === STEP 3: Bootstrap clean pip for Python 3.12 ===
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.12 get-pip.py --no-cache-dir && \
    rm get-pip.py && \
    python -m pip install --no-cache-dir --upgrade pip

# Verify Python/pip installation
RUN python --version && pip --version

# === STEP 4: Set Working Directory ===
WORKDIR /app

# === STEP 5: Copy requirements FIRST (cache optimization) ===
COPY requirements.txt .

# === STEP 6: INSTALL PYTHON DEPENDENCIES (CRITICAL FIX) ===
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# === STEP 7: Copy Project Code ===
COPY t_trace/ ./t_trace/
COPY config.yml ./config.yml
COPY experiments/ ./experiments/

# === STEP 8: Add Project to PYTHONPATH ===
ENV PYTHONPATH=/app:${PYTHONPATH}

# === STEP 9: Create Non-Root User (Security) ===
RUN useradd -m -u 1000 mtrace && \
    chown -R mtrace:mtrace /app
USER mtrace

# === STEP 10: Default Command ===
CMD ["python", "-c", "print('M-TRACE Container Ready')"]