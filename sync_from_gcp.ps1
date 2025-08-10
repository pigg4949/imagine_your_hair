# GCP 서버에서 로컬로 동기화 스크립트 (PowerShell)
# 사용법: .\sync_from_gcp.ps1

param(
    [Parameter(Mandatory=$true)]
    [string]$ServerIP,
    
    [Parameter(Mandatory=$true)]
    [string]$Username,
    
    [Parameter(Mandatory=$false)]
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_rsa",
    
    [Parameter(Mandatory=$false)]
    [string]$RemotePath = "/home/$Username/HairSwap",
    
    [Parameter(Mandatory=$false)]
    [string]$SyncType = "all"  # "all", "outputs", "logs", "config"
)

Write-Host "🔄 GCP 서버에서 로컬로 동기화 시작..." -ForegroundColor Green

# 동기화 타입별 설정
$syncConfigs = @{
    "all" = @(".")
    "outputs" = @("outputs", "logs", "app.log")
    "logs" = @("app.log", "*.log")
    "config" = @("*.py", "*.txt", "*.md", "templates", "requirements.txt")
}

$syncItems = $syncConfigs[$SyncType]

Write-Host "📋 동기화 항목: $SyncType" -ForegroundColor Yellow
foreach ($item in $syncItems) {
    Write-Host "  - $item" -ForegroundColor White
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
    exit 1
}

# 동기화 실행
Write-Host "📥 파일 동기화 중..." -ForegroundColor Yellow
try {
    foreach ($item in $syncItems) {
        Write-Host "  📤 $item 동기화 중..." -ForegroundColor Cyan
        
        if ($item -eq ".") {
            # 전체 프로젝트 동기화 (Git 제외)
            ssh -i $KeyPath "$Username@$ServerIP" "cd $RemotePath && tar --exclude='.git' --exclude='venv' --exclude='__pycache__' -czf - ." | tar -xzf - --strip-components=0
        } else {
            # 특정 파일/폴더 동기화
            scp -i $KeyPath -r "$Username@$ServerIP:$RemotePath/$item" .
        }
        
        Write-Host "  ✅ $item 동기화 완료" -ForegroundColor Green
    }
    
    Write-Host "🎉 모든 동기화 완료!" -ForegroundColor Green
    
} catch {
    Write-Host "❌ 동기화 실패: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Git 상태 확인
Write-Host "📊 Git 상태 확인 중..." -ForegroundColor Yellow
try {
    $gitStatus = git status --porcelain
    if ($gitStatus) {
        Write-Host "📝 변경된 파일들:" -ForegroundColor Cyan
        $gitStatus | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
        
        $commitChoice = Read-Host "변경사항을 커밋하시겠습니까? (y/n)"
        if ($commitChoice -eq "y" -or $commitChoice -eq "Y") {
            $commitMessage = Read-Host "커밋 메시지를 입력하세요"
            git add .
            git commit -m "🔄 GCP 서버에서 동기화: $commitMessage"
            git push origin dev
            Write-Host "✅ 변경사항이 커밋되고 푸시되었습니다." -ForegroundColor Green
        }
    } else {
        Write-Host "✅ 변경사항이 없습니다." -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  Git 상태 확인 중 오류: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📋 사용법 안내:" -ForegroundColor Cyan
Write-Host "  전체 동기화: .\sync_from_gcp.ps1 -ServerIP '34.123.45.67' -Username 'your-username' -SyncType 'all'" -ForegroundColor Gray
Write-Host "  출력만 동기화: .\sync_from_gcp.ps1 -ServerIP '34.123.45.67' -Username 'your-username' -SyncType 'outputs'" -ForegroundColor Gray
Write-Host "  로그만 동기화: .\sync_from_gcp.ps1 -ServerIP '34.123.45.67' -Username 'your-username' -SyncType 'logs'" -ForegroundColor Gray
Write-Host ""
