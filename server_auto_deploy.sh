#!/bin/bash

# 서버 자동 배포 스크립트
# 사용법: bash server_auto_deploy.sh

# 설정
PROJECT_DIR="/home/$USER/HairSwap"
GIT_BRANCH="dev"
SERVICE_NAME="hairswap"
LOG_FILE="$PROJECT_DIR/deploy.log"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

# 프로젝트 디렉토리로 이동
cd "$PROJECT_DIR" || {
    error "프로젝트 디렉토리를 찾을 수 없습니다: $PROJECT_DIR"
    exit 1
}

log "🚀 자동 배포 시작"

# Git 상태 확인
if [ ! -d ".git" ]; then
    error "Git 저장소가 아닙니다"
    exit 1
fi

# 현재 브랜치 확인
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$GIT_BRANCH" ]; then
    warning "현재 브랜치가 $GIT_BRANCH가 아닙니다: $CURRENT_BRANCH"
    git checkout "$GIT_BRANCH" || {
        error "브랜치 전환 실패"
        exit 1
    }
fi

# 원격 변경사항 확인
log "📡 원격 변경사항 확인 중..."
git fetch origin "$GIT_BRANCH"

# 로컬과 원격 비교
LOCAL_COMMIT=$(git rev-parse HEAD)
REMOTE_COMMIT=$(git rev-parse origin/$GIT_BRANCH)

if [ "$LOCAL_COMMIT" = "$REMOTE_COMMIT" ]; then
    info "변경사항이 없습니다"
    exit 0
fi

log "🔄 변경사항 감지됨"
log "로컬: $LOCAL_COMMIT"
log "원격: $REMOTE_COMMIT"

# 기존 서비스 중지
log "🛑 기존 서비스 중지 중..."
if pgrep -f "python.*app.py" > /dev/null; then
    pkill -f "python.*app.py"
    sleep 2
    if pgrep -f "python.*app.py" > /dev/null; then
        warning "서비스가 완전히 중지되지 않았습니다. 강제 종료..."
        pkill -9 -f "python.*app.py"
    fi
fi

# 변경사항 가져오기
log "📥 변경사항 가져오기..."
git pull origin "$GIT_BRANCH" || {
    error "Git pull 실패"
    exit 1
}

# 가상환경 활성화
if [ -d "venv" ]; then
    log "📦 가상환경 활성화..."
    source venv/bin/activate
else
    error "가상환경을 찾을 수 없습니다"
    exit 1
fi

# 의존성 업데이트 확인
if [ -f "requirements.txt" ]; then
    log "📦 의존성 업데이트 확인..."
    pip install -r requirements.txt --quiet
fi

# 필요한 디렉토리 생성
log "📁 필요한 디렉토리 생성..."
mkdir -p outputs
mkdir -p templates
mkdir -p logs

# 서비스 시작
log "🚀 서비스 시작..."
nohup python app.py > app.log 2>&1 &
SERVICE_PID=$!

# 서비스 시작 확인
sleep 3
if kill -0 $SERVICE_PID 2>/dev/null; then
    log "✅ 서비스가 성공적으로 시작되었습니다 (PID: $SERVICE_PID)"
    
    # 헬스 체크
    sleep 2
    if curl -s http://localhost:8000/health > /dev/null; then
        log "✅ 헬스 체크 통과"
    else
        warning "헬스 체크 실패 (서비스가 아직 완전히 시작되지 않았을 수 있습니다)"
    fi
    
else
    error "서비스 시작 실패"
    exit 1
fi

# GPU 상태 확인
if command -v nvidia-smi > /dev/null; then
    log "🎮 GPU 상태 확인..."
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits
fi

log "🎉 배포 완료!"

# 배포 정보 출력
echo ""
echo "📊 배포 정보:"
echo "  - 서비스 PID: $SERVICE_PID"
echo "  - 로그 파일: app.log"
echo "  - 웹 인터페이스: http://localhost:8000"
echo "  - API 문서: http://localhost:8000/docs"
echo "  - 헬스 체크: http://localhost:8000/health"
echo ""

# 서비스 모니터링 시작 (선택사항)
if [ "$1" = "--monitor" ]; then
    log "📊 서비스 모니터링 시작..."
    while kill -0 $SERVICE_PID 2>/dev/null; do
        echo -n "."
        sleep 10
    done
    error "서비스가 종료되었습니다"
fi
