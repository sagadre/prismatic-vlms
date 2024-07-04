# ===
# Prismatic VLM Sagemaker Dockerfile
#   => Base Image :: Python 3.10 & Pytorch 2.2.0
# ===
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker

# Sane Defaults
RUN apt-get update
RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    docker.io \
    ffmpeg \
    git \
    htop \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    libgl1 \
    libopenexr-dev \
    mesa-utils \
    freeglut3-dev \
    libsdl2-2.0-0 \
    python-pygame

# IMPORTANT :: Uninstall & Reinstall Torch (Sagemaker CPU Core Bug)
RUN pip install --upgrade pip
RUN pip uninstall -y torch
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install Prismatic Python Dependencies (`pip`) + Sagemaker
RUN pip install \
    accelerate>=0.25.0 \
    draccus==0.8.0 \
    einops \
    huggingface_hub==0.22.2 \
    jsonlines \
    matplotlib \
    protobuf \
    rich \
    sentencepiece==0.1.99 \
    timm==0.9.10 \
    tokenizers==0.19.1 \
    transformers==4.40.1 \
    wandb \
    sagemaker-training

# Flash Attention 2 Installation
RUN pip install packaging ninja
RUN pip install flash-attn==2.5.5 --no-build-isolation


# Set Sagemaker Environment Variables =>> Define `pretrain.py` as entrypoint!
ENV PATH="/opt/ml/code:${PATH}"

ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=/opt/ml/code/scripts/pretrain.py

# Copy Working Directory to `/opt/ml/code`
COPY . /opt/ml/code/
