#!/bin/bash

# GCP HairGen/SD3 + DreamBooth 환경 설정 스크립트
# Ubuntu 22.04 LTS 기준

echo "🚀 GCP HairGen/SD3 + DreamBooth 환경 설정 시작..."

# 시스템 업데이트
echo "📦 시스템 업데이트 중..."
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
echo "🔧 필수 패키지 설치 중..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libc6-dev \
    libstdc++6 \
    libgcc1 \
    libgomp1 \
    libatomic1 \
    libc6 \
    libstdc++6

# CUDA 및 cuDNN 설치 (T4 GPU용)
echo "🎮 CUDA 및 cuDNN 설치 중..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-0

# Python 환경 설정
echo "🐍 Python 환경 설정 중..."
python3 -m venv ~/hairgen_env
source ~/hairgen_env/bin/activate

# PyTorch 설치 (CUDA 12.0 지원)
echo "🔥 PyTorch 설치 중..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 필수 Python 패키지 설치
echo "📚 Python 패키지 설치 중..."
pip install \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    xformers \
    opencv-python \
    pillow \
    numpy \
    scipy \
    matplotlib \
    tqdm \
    wandb \
    tensorboard \
    gradio \
    fastapi \
    uvicorn \
    python-multipart

# HairGen 관련 패키지
echo "💇 HairGen 패키지 설치 중..."
pip install \
    mediapipe \
    face-recognition \
    dlib \
    opencv-contrib-python

# 디렉토리 구조 생성
echo "📁 디렉토리 구조 생성 중..."
mkdir -p ~/hairgen_project/{data,models,outputs,checkpoints,logs}
mkdir -p ~/sd3_project/{data,models,outputs,checkpoints,logs}

# 환경 변수 설정
echo "🔧 환경 변수 설정 중..."
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
echo 'export PYTHONPATH="${PYTHONPATH}:~/hairgen_project"' >> ~/.bashrc
echo 'export PYTHONPATH="${PYTHONPATH}:~/sd3_project"' >> ~/.bashrc

# GPU 확인
echo "🎮 GPU 상태 확인 중..."
nvidia-smi

# 디스크 사용량 확인
echo "💾 디스크 사용량 확인 중..."
df -h

echo "✅ GCP 환경 설정 완료!"
echo "🎯 다음 단계:"
echo "1. source ~/hairgen_env/bin/activate"
echo "2. cd ~/hairgen_project"
echo "3. git clone [your-repository]"
echo "4. python setup.py install" 