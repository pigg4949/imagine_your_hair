# 💇‍♀️ AI 헤어스타일 전송 서비스

AI 기반 헤어스타일 전송 웹 서비스입니다. 사용자가 원본 이미지와 원하는 헤어스타일을 가진 참조 이미지를 업로드하면, 자연스럽게 헤어스타일을 합성해주는 서비스입니다.

## 🚀 주요 기능

- **실시간 헤어스타일 전송**: Stable Diffusion과 Face Parsing을 활용한 고품질 헤어스타일 합성
- **웹 인터페이스**: 직관적이고 사용하기 쉬운 웹 UI
- **GPU 가속**: NVIDIA GPU를 활용한 빠른 처리 속도
- **REST API**: 외부 시스템과의 연동을 위한 API 제공
- **예시 이미지**: 테스트용 예시 이미지 제공

## 🛠️ 기술 스택

- **Backend**: FastAPI, Python 3.8+
- **AI/ML**: PyTorch, Stable Diffusion, BiSeNet (Face Parsing)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **GPU**: NVIDIA CUDA 12.0+
- **Cloud**: Google Cloud Platform (GCP)

## 📋 요구사항

### 시스템 요구사항
- **OS**: Ubuntu 22.04 LTS (권장)
- **CPU**: 8 vCPU 이상
- **RAM**: 30GB 이상
- **GPU**: NVIDIA T4 이상 (CUDA 지원)
- **Storage**: 50GB 이상

### 소프트웨어 요구사항
- Python 3.8+
- CUDA 12.0+
- cuDNN 8.0+

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/HairSwap.git
cd HairSwap
```

### 2. 가상환경 생성 및 활성화
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 모델 다운로드
```bash
# Face Parsing 모델 다운로드
mkdir -p inference/face_parsing/res/cp
wget -O inference/face_parsing/res/cp/model_final_diss.pth \
     "https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/model_final_diss.pth"
```

### 5. 서비스 실행
```bash
python app.py
```

### 6. 웹 브라우저에서 접속
```
http://localhost:8000
```

## 🌐 GCP 배포

GCP에서 GPU를 활용한 서비스 배포는 `GCP_설정_가이드.txt` 파일을 참조하세요.

### 주요 단계:
1. GCP VM 인스턴스 생성 (n1-standard-8 + NVIDIA T4)
2. CUDA 및 Python 환경 설정
3. 모델 다운로드 및 설정
4. 서비스 실행 및 방화벽 설정

## 📖 API 문서

### 주요 엔드포인트

#### 1. 헬스 체크
```http
GET /health
```

#### 2. 이미지 업로드
```http
POST /upload
Content-Type: multipart/form-data

file: [이미지 파일]
```

#### 3. 헤어스타일 전송
```http
POST /transfer
Content-Type: application/x-www-form-urlencoded

source_image: [업로드된 원본 이미지 파일명]
reference_image: [업로드된 참조 이미지 파일명]
prompt: [긍정 프롬프트] (선택사항)
negative_prompt: [부정 프롬프트] (선택사항)
```

#### 4. 결과 다운로드
```http
GET /download/{filename}
```

#### 5. 예시 이미지 목록
```http
GET /examples
```

### API 문서 접속
- Swagger UI: `http://your-domain:8000/docs`
- ReDoc: `http://your-domain:8000/redoc`

## 🏗️ 프로젝트 구조

```
HairSwap/
├── app.py                          # FastAPI 메인 애플리케이션
├── requirements.txt                # Python 의존성
├── README.md                      # 프로젝트 문서
├── GCP_설정_가이드.txt             # GCP 배포 가이드
├── templates/
│   └── index.html                 # 웹 인터페이스
├── hair_style_transfer/
│   ├── improved_hair_transfer.py  # 개선된 헤어스타일 전송 모델
│   ├── segmentation_based_hair_transfer.py
│   ├── simple_hair_transfer.py
│   └── utils/                     # 유틸리티 함수들
├── inference/
│   └── face_parsing/              # Face Parsing 모델
├── images/
│   ├── original/                  # 예시 이미지들
│   ├── only_hair/                 # 헤어만 추출된 이미지
│   └── only_hair_mask/            # 헤어 마스크 이미지
├── outputs/                       # 생성된 결과 이미지
└── venv/                         # Python 가상환경
```

## 🔧 설정

### 환경 변수
```bash
# GPU 설정
export CUDA_VISIBLE_DEVICES=0

# Python 경로 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 설정 파일
주요 설정은 `app.py` 파일에서 수정할 수 있습니다:
- 포트 번호 (기본: 8000)
- GPU/CPU 사용 설정
- 모델 경로 설정

## 📊 성능 최적화

### GPU 메모리 최적화
- `xformers` 라이브러리 사용으로 메모리 효율성 향상
- 배치 크기 조정으로 메모리 사용량 제어
- 모델 양자화 (필요시)

### 처리 속도 최적화
- CUDA 가속 활용
- 모델 캐싱
- 비동기 처리

## 🐛 문제 해결

### 일반적인 문제들

#### 1. GPU 인식 문제
```bash
# GPU 상태 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

#### 2. 모델 로드 실패
```bash
# 모델 파일 존재 확인
ls -la inference/face_parsing/res/cp/

# 수동 모델 테스트
python -c "from improved_hair_transfer import ImprovedHairTransfer; model = ImprovedHairTransfer()"
```

#### 3. 메모리 부족
```bash
# 메모리 사용량 확인
free -h
nvidia-smi

# 스왑 파일 생성 (필요시)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🙏 감사의 말

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - 이미지 생성 모델
- [Face Parsing](https://github.com/zllrunning/face-parsing.PyTorch) - 얼굴 분할 모델
- [FastAPI](https://fastapi.tiangolo.com/) - 웹 프레임워크
- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크

## 📞 지원

문제가 발생하거나 질문이 있으시면:
- GitHub Issues를 통해 문의
- 이메일: your-email@example.com

## 🔄 업데이트 로그

### v1.0.0 (2024-01-XX)
- 초기 버전 릴리즈
- 기본 헤어스타일 전송 기능
- 웹 인터페이스 구현
- GCP 배포 가이드 추가
