# GCP 서버 업로드 스크립트 (PowerShell)
# 사용법: .\upload_to_gcp.ps1

param(
    [Parameter(Mandatory=$true)]
    [string]$ServerIP,
    
    [Parameter(Mandatory=$true)]
    [string]$Username,
    
    [Parameter(Mandatory=$false)]
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_rsa",
    
    [Parameter(Mandatory=$false)]
    [string]$RemotePath = "/home/$Username/HairSwap"
)

Write-Host "🚀 GCP 서버 업로드 시작..." -ForegroundColor Green

# 필수 파일들 확인
$requiredFiles = @(
    "app.py",
    "requirements.txt",
    "hair_style_transfer\improved_hair_transfer.py",
    "templates\index.html",
    "images\original",
    "inference\face_parsing"
)

Write-Host "📋 필수 파일 확인 중..." -ForegroundColor Yellow
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✅ $file" -ForegroundColor Green
    } else {
        Write-Host "❌ $file (누락됨)" -ForegroundColor Red
    }
}

# SSH 키 확인
if (-not (Test-Path $KeyPath)) {
    Write-Host "⚠️  SSH 키를 찾을 수 없습니다: $KeyPath" -ForegroundColor Yellow
    Write-Host "🔑 SSH 키를 생성하거나 경로를 확인해주세요." -ForegroundColor Yellow
    exit 1
}

# 서버 연결 테스트
Write-Host "🔗 서버 연결 테스트 중..." -ForegroundColor Yellow
try {
    $testConnection = ssh -i $KeyPath -o ConnectTimeout=10 -o BatchMode=yes "$Username@$ServerIP" "echo 'Connection successful'"
    if ($testConnection -eq "Connection successful") {
        Write-Host "✅ 서버 연결 성공!" -ForegroundColor Green
    } else {
        throw "연결 실패"
    }
} catch {
    Write-Host "❌ 서버 연결 실패: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "🔧 다음을 확인해주세요:" -ForegroundColor Yellow
    Write-Host "   1. 서버 IP 주소가 올바른지" -ForegroundColor White
    Write-Host "   2. SSH 키가 서버에 등록되었는지" -ForegroundColor White
    Write-Host "   3. 방화벽에서 SSH 포트(22)가 열려있는지" -ForegroundColor White
    exit 1
}

# 원격 디렉토리 생성
Write-Host "📁 원격 디렉토리 생성 중..." -ForegroundColor Yellow
ssh -i $KeyPath "$Username@$ServerIP" "mkdir -p $RemotePath"

# 파일 업로드
Write-Host "📤 파일 업로드 중..." -ForegroundColor Yellow
try {
    # 전체 프로젝트 폴더 업로드
    scp -i $KeyPath -r . "$Username@$ServerIP:$RemotePath"
    Write-Host "✅ 파일 업로드 완료!" -ForegroundColor Green
} catch {
    Write-Host "❌ 파일 업로드 실패: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# 서버에서 환경 설정
Write-Host "🔧 서버 환경 설정 중..." -ForegroundColor Yellow
$setupCommands = @"
cd $RemotePath
echo '📦 Python 가상환경 생성 중...'
python3 -m venv venv
source venv/bin/activate
echo '📥 의존성 설치 중...'
pip install -r requirements.txt
echo '📁 필요한 디렉토리 생성 중...'
mkdir -p outputs
mkdir -p templates
echo '✅ 환경 설정 완료!'
"@

try {
    ssh -i $KeyPath "$Username@$ServerIP" "$setupCommands"
    Write-Host "✅ 서버 환경 설정 완료!" -ForegroundColor Green
} catch {
    Write-Host "⚠️  환경 설정 중 오류 발생: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "🔧 수동으로 설정해주세요." -ForegroundColor Yellow
}

# 서비스 실행 명령어 안내
Write-Host ""
Write-Host "🎉 업로드 완료!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 다음 단계:" -ForegroundColor Cyan
Write-Host "1. SSH로 서버에 접속:" -ForegroundColor White
Write-Host "   ssh -i $KeyPath $Username@$ServerIP" -ForegroundColor Gray
Write-Host ""
Write-Host "2. 프로젝트 디렉토리로 이동:" -ForegroundColor White
Write-Host "   cd $RemotePath" -ForegroundColor Gray
Write-Host ""
Write-Host "3. 서비스 실행:" -ForegroundColor White
Write-Host "   python app.py" -ForegroundColor Gray
Write-Host "   또는" -ForegroundColor White
Write-Host "   bash run_service.sh" -ForegroundColor Gray
Write-Host ""
Write-Host "4. 웹 브라우저에서 접속:" -ForegroundColor White
Write-Host "   http://$ServerIP:8000" -ForegroundColor Gray
Write-Host ""
