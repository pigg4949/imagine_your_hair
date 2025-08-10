#!/bin/bash

# Hair Style Transfer Service 실행 스크립트
# GCP 환경에서 사용

echo "🚀 Hair Style Transfer Service 시작 중..."

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 활성화 (존재하는 경우)
if [ -d "venv" ]; then
    echo "📦 가상환경 활성화 중..."
    source venv/bin/activate
elif [ -d "$HOME/hairgen_env" ]; then
    echo "📦 전역 가상환경 활성화 중..."
    source $HOME/hairgen_env/bin/activate
fi

# GPU 상태 확인
echo "🎮 GPU 상태 확인 중..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "⚠️  NVIDIA GPU가 감지되지 않았습니다. CPU 모드로 실행됩니다."
fi

# 필요한 디렉토리 생성
echo "📁 필요한 디렉토리 생성 중..."
mkdir -p outputs
mkdir -p templates

# 환경 변수 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# 서비스 실행
echo "🌟 서비스 시작 중..."
echo "📍 웹 인터페이스: http://localhost:8000"
echo "📍 API 문서: http://localhost:8000/docs"
echo "📍 헬스 체크: http://localhost:8000/health"
echo ""
echo "서비스를 중지하려면 Ctrl+C를 누르세요."
echo ""

# 로그 파일로 출력하면서 백그라운드 실행
python app.py 2>&1 | tee app.log
