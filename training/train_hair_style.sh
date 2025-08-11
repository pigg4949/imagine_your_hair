#!/bin/bash

# 헤어스타일 DreamBooth 학습 스크립트
# 사용법: ./train_hair_style.sh [데이터_디렉토리] [헤어스타일_토큰] [학습_스텝]

set -e  # 오류 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 인자 확인
if [ $# -lt 2 ]; then
    echo "사용법: $0 <데이터_디렉토리> <헤어스타일_토큰> [학습_스텝]"
    echo "예시: $0 ./raw_images hst 1000"
    exit 1
fi

DATA_DIR="$1"
HAIR_TOKEN="$2"
TRAIN_STEPS="${3:-1000}"  # 기본값 1000

# 디렉토리 확인
if [ ! -d "$DATA_DIR" ]; then
    log_error "데이터 디렉토리를 찾을 수 없습니다: $DATA_DIR"
    exit 1
fi

log_info "헤어스타일 DreamBooth 학습 시작"
log_info "데이터 디렉토리: $DATA_DIR"
log_info "헤어스타일 토큰: $HAIR_TOKEN"
log_info "학습 스텝: $TRAIN_STEPS"

# 1단계: 데이터 전처리
log_info "1단계: 데이터 전처리 시작"
python training/data_preparation.py \
    --input_dir "$DATA_DIR" \
    --output_dir "training_data" \
    --target_size 512 512 \
    --min_face_size 100 \
    --quality_threshold 0.7 \
    --validation_ratio 0.2

if [ $? -ne 0 ]; then
    log_error "데이터 전처리 실패"
    exit 1
fi
log_success "데이터 전처리 완료"

# 학습 데이터 개수 확인
TRAINING_COUNT=$(ls training_data/*.jpg 2>/dev/null | wc -l)
if [ "$TRAINING_COUNT" -lt 10 ]; then
    log_warning "학습 데이터가 부족합니다: $TRAINING_COUNT개"
    log_warning "최소 10개 이상의 이미지가 필요합니다"
    read -p "계속 진행하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2단계: DreamBooth 학습
log_info "2단계: DreamBooth 학습 시작"
log_info "예상 소요 시간: $((TRAIN_STEPS / 100))분"

python training/hair_style_dreambooth.py \
    --training_data_dir "training_data" \
    --output_dir "trained_hair_models" \
    --hair_style_token "$HAIR_TOKEN" \
    --max_train_steps "$TRAIN_STEPS" \
    --learning_rate 1e-6 \
    --use_lora

if [ $? -ne 0 ]; then
    log_error "DreamBooth 학습 실패"
    exit 1
fi
log_success "DreamBooth 학습 완료"

# 3단계: 결과 확인
log_info "3단계: 학습 결과 확인"

# 모델 파일 확인
if [ -f "trained_hair_models/hair_style_config.json" ]; then
    log_success "모델 설정 파일 생성됨"
    echo "=== 학습 설정 ==="
    cat "trained_hair_models/hair_style_config.json" | python -m json.tool
else
    log_error "모델 설정 파일을 찾을 수 없습니다"
fi

# LoRA 파일 확인
LORA_FILES=$(ls trained_hair_models/*.safetensors 2>/dev/null | wc -l)
if [ "$LORA_FILES" -gt 0 ]; then
    log_success "LoRA 모델 파일 생성됨: $LORA_FILES개"
else
    log_warning "LoRA 모델 파일을 찾을 수 없습니다"
fi

# 4단계: 사용법 안내
log_info "4단계: 학습 완료!"
echo
echo "=== 사용법 ==="
echo "학습된 모델을 사용하려면:"
echo "프롬프트 예시:"
echo "  - '$HAIR_TOKEN person with long blonde hair'"
echo "  - '$HAIR_TOKEN person with short black hair'"
echo "  - '$HAIR_TOKEN person with curly brown hair'"
echo
echo "모델 경로: trained_hair_models/"
echo "설정 파일: trained_hair_models/hair_style_config.json"
echo

# 5단계: 테스트 스크립트 생성
log_info "5단계: 테스트 스크립트 생성"

cat > "test_trained_model.py" << EOF
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor
import os

def test_trained_model():
    """학습된 모델 테스트"""
    
    # 기본 모델 로드
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # LoRA 가중치 로드
    if os.path.exists("trained_hair_models"):
        pipe.unet.load_attn_procs("trained_hair_models")
        print("✅ LoRA 가중치 로드 완료")
    
    # GPU 사용
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("✅ GPU 사용")
    
    # 테스트 프롬프트
    test_prompts = [
        "a photo of $HAIR_TOKEN person with long blonde hair",
        "a photo of $HAIR_TOKEN person with short black hair",
        "a photo of $HAIR_TOKEN person with curly brown hair",
    ]
    
    # 이미지 생성
    for i, prompt in enumerate(test_prompts):
        print(f"생성 중: {prompt}")
        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save(f"test_result_{i}.png")
        print(f"✅ 이미지 저장: test_result_{i}.png")

if __name__ == "__main__":
    test_trained_model()
EOF

chmod +x test_trained_model.py
log_success "테스트 스크립트 생성됨: test_trained_model.py"

echo
log_success "🎉 헤어스타일 DreamBooth 학습이 완료되었습니다!"
echo
echo "다음 단계:"
echo "1. python test_trained_model.py  # 학습된 모델 테스트"
echo "2. 생성된 이미지 확인"
echo "3. 필요시 재학습 또는 파라미터 조정"
