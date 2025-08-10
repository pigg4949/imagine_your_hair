# 헤어스타일 변경 AI 모델

ControlNet + Inpainting을 사용하여 자연스러운 헤어스타일 변경을 수행하는 AI 모델입니다.

## 기능

- 원본 얼굴 유지하면서 헤어스타일만 변경
- CPU 환경에서 최적화된 성능
- ControlNet을 통한 정밀한 헤어 구조 제어
- Inpainting을 통한 자연스러운 합성

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 기본 사용법

```python
from hair_style_transfer import HairStyleTransfer

# 모델 초기화
hair_transfer = HairStyleTransfer(device="cpu")

# 헤어스타일 변경
success = hair_transfer.transfer_hair_style(
    source_image_path="path/to/source.jpg",
    reference_hair_path="path/to/hair.png",
    output_path="path/to/output.jpg"
)
```

### 2. 고급 옵션

```python
success = hair_transfer.transfer_hair_style(
    source_image_path="source.jpg",
    reference_hair_path="hair.png",
    output_path="output.jpg",
    prompt="long curly hair, natural, high quality",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=30,
    guidance_scale=8.0
)
```

### 3. 배치 처리

```python
import os

source_dir = "images/original"
reference_dir = "images/only_hair"
output_dir = "output"

for source_file in os.listdir(source_dir):
    if source_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        source_path = os.path.join(source_dir, source_file)
        reference_path = os.path.join(reference_dir, "reference_hair.png")
        output_path = os.path.join(output_dir, f"transferred_{source_file}")

        hair_transfer.transfer_hair_style(
            source_image_path=source_path,
            reference_hair_path=reference_path,
            output_path=output_path
        )
```

## 매개변수

- `source_image_path`: 원본 이미지 경로
- `reference_hair_path`: 참조 헤어 이미지 경로 (투명 배경 PNG 권장)
- `output_path`: 출력 이미지 경로
- `prompt`: 긍정적 프롬프트 (기본값: "natural hair, high quality, detailed")
- `negative_prompt`: 부정적 프롬프트 (기본값: "blurry, low quality, distorted")
- `num_inference_steps`: 추론 스텝 수 (기본값: 20, 높을수록 품질 향상)
- `guidance_scale`: 가이던스 스케일 (기본값: 7.5, 높을수록 프롬프트 준수도 증가)

## 폴더 구조

```
hair_style_transfer/
├── hair_style_transfer.py    # 메인 모델
├── requirements.txt          # 의존성 패키지
├── README.md                # 사용법 설명
├── models/                  # 모델 파일들
├── utils/                   # 유틸리티 함수들
└── output/                  # 출력 결과
```

## 주의사항

1. **메모리 사용량**: CPU 환경에서 실행 시 충분한 RAM이 필요합니다 (최소 8GB 권장)
2. **처리 시간**: 이미지당 1-3분 정도 소요될 수 있습니다
3. **이미지 크기**: 자동으로 512x512로 리사이즈됩니다
4. **참조 헤어**: 투명 배경의 PNG 파일을 사용하는 것을 권장합니다

## 문제 해결

### 모델 로드 실패

- 인터넷 연결 확인
- 충분한 디스크 공간 확보
- Python 버전 확인 (3.8 이상 권장)

### 메모리 부족

- `num_inference_steps`를 줄여보세요
- 더 작은 이미지 크기 사용
- 다른 프로그램 종료

### 품질 개선

- `num_inference_steps` 증가 (20 → 30)
- `guidance_scale` 조정 (7.5 → 8.0)
- 더 구체적인 프롬프트 사용
