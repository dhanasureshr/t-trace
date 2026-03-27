# Base Image: CUDA 12.3 + cuDNN 9 + Ubuntu 22.04 (Matches Experimental Plan)
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Prevent Interactive Prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System Dependencies (Including libsnappy for compression engine)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    libsnappy-dev \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create Non-Root User (Matches Host UID/GID to avoid permission issues on mounted volumes)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} mtrace && \
    useradd -u ${USER_ID} -g mtrace -m -s /bin/bash mtrace

# Set Working Directory
WORKDIR /app/mtrace

# Copy Requirements First (Layer Caching)
COPY requirements.txt .

# Create Virtual Environment & Install Dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Project Code
COPY --chown=mtrace:mtrace . .


# Set executable permissions INSIDE the image (not on mounted volumes)
RUN chmod +x t_trace/experiments/run_all_experiments.sh

# Switch to Non-Root User
USER mtrace

# Default Command
CMD ["/bin/bash"]