# Use a known working PyTorch CUDA image
FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git libsdl2-2.0-0 libsdl2-image-2.0-0 libsdl2-ttf-2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "pokemon_blue_agent.py"]
