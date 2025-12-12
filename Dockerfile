# Use the official NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV SDL_VIDEODRIVER=dummy
ENV SDL_AUDIODRIVER=dummy

# Install Python and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3 \
        python3-pip \
        python3-venv \
        libsdl2-2.0-0 \
        libsdl2-image-2.0-0 \
        libsdl2-ttf-2.0-0 \
        libjpeg-dev \
        zlib1g-dev \
    && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu122 torch==2.6.0 && \
    python3 -m pip install -r requirements.txt

COPY . .

# Declare env_state as a volume for persistence and host access
VOLUME ["/workspace/env_state"]

ENTRYPOINT ["python3", "pokemon_blue_agent.py"]
