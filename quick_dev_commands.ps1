# 빠른 개발 명령어 모음 (PowerShell)
# 사용법: .\quick_dev_commands.ps1

param(
    [Parameter(Mandatory=$true)]
    [string]$ServerIP,
    
    [Parameter(Mandatory=$true)]
    [string]$Username,
    
    [Parameter(Mandatory=$false)]
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_rsa"
)

function Show-Menu {
    Clear-Host
    Write-Host "🚀 HairSwap 개발 명령어 모음" -ForegroundColor Green
    Write-Host "=================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "1.  📡 서버 상태 확인" -ForegroundColor Cyan
    Write-Host "2.  🎮 GPU 상태 확인" -ForegroundColor Cyan
    Write-Host "3.  🚀 서비스 시작" -ForegroundColor Cyan
    Write-Host "4.  🛑 서비스 중지" -ForegroundColor Cyan
    Write-Host "5.  📊 로그 확인" -ForegroundColor Cyan
    Write-Host "6.  🔄 Git 동기화" -ForegroundColor Cyan
    Write-Host "7.  🧪 테스트 실행" -ForegroundColor Cyan
    Write-Host "8.  📁 결과물 확인" -ForegroundColor Cyan
    Write-Host "9.  🔧 서버 접속" -ForegroundColor Cyan
    Write-Host "10. 📤 결과물 다운로드" -ForegroundColor Cyan
    Write-Host "11. 🎯 AI 모델 상태 확인" -ForegroundColor Cyan
    Write-Host "12. 💾 메모리 사용량 확인" -ForegroundColor Cyan
    Write-Host "0.  ❌ 종료" -ForegroundColor Red
    Write-Host ""
}

function Execute-RemoteCommand {
    param([string]$Command, [string]$Description)
    Write-Host "🔄 $Description 중..." -ForegroundColor Yellow
    try {
        $result = ssh -i $KeyPath "$Username@$ServerIP" "$Command"
        Write-Host "✅ $Description 완료:" -ForegroundColor Green
        Write-Host $result -ForegroundColor White
    } catch {
        Write-Host "❌ $Description 실패: $($_.Exception.Message)" -ForegroundColor Red
    }
    Write-Host ""
    Read-Host "계속하려면 Enter를 누르세요"
}

function Show-Logs {
    Write-Host "📊 로그 확인 옵션:" -ForegroundColor Cyan
    Write-Host "1. 실시간 로그 (Ctrl+C로 종료)" -ForegroundColor White
    Write-Host "2. 최근 50줄" -ForegroundColor White
    Write-Host "3. 에러 로그만" -ForegroundColor White
    Write-Host "4. 뒤로 가기" -ForegroundColor White
    
    $choice = Read-Host "선택하세요 (1-4)"
    
    switch ($choice) {
        "1" { 
            Write-Host "🔄 실시간 로그 확인 중... (Ctrl+C로 종료)" -ForegroundColor Yellow
            ssh -i $KeyPath "$Username@$ServerIP" "tail -f /home/$Username/HairSwap/app.log"
        }
        "2" { 
            Execute-RemoteCommand "tail -n 50 /home/$Username/HairSwap/app.log" "최근 50줄 로그 확인"
        }
        "3" { 
            Execute-RemoteCommand "grep -i error /home/$Username/HairSwap/app.log | tail -20" "에러 로그 확인"
        }
        "4" { return }
    }
}

function Download-Results {
    Write-Host "📤 결과물 다운로드 옵션:" -ForegroundColor Cyan
    Write-Host "1. 전체 outputs 폴더" -ForegroundColor White
    Write-Host "2. 최근 생성된 파일만" -ForegroundColor White
    Write-Host "3. 특정 파일" -ForegroundColor White
    Write-Host "4. 뒤로 가기" -ForegroundColor White
    
    $choice = Read-Host "선택하세요 (1-4)"
    
    switch ($choice) {
        "1" { 
            Write-Host "📥 전체 outputs 폴더 다운로드 중..." -ForegroundColor Yellow
            scp -i $KeyPath -r "$Username@$ServerIP:/home/$Username/HairSwap/outputs/" .
            Write-Host "✅ 다운로드 완료!" -ForegroundColor Green
        }
        "2" { 
            Write-Host "📥 최근 파일 다운로드 중..." -ForegroundColor Yellow
            ssh -i $KeyPath "$Username@$ServerIP" "find /home/$Username/HairSwap/outputs/ -type f -mtime -1 -ls"
            scp -i $KeyPath "$Username@$ServerIP:/home/$Username/HairSwap/outputs/*" ./outputs/
            Write-Host "✅ 다운로드 완료!" -ForegroundColor Green
        }
        "3" { 
            $filename = Read-Host "파일명을 입력하세요"
            Write-Host "📥 $filename 다운로드 중..." -ForegroundColor Yellow
            scp -i $KeyPath "$Username@$ServerIP:/home/$Username/HairSwap/outputs/$filename" .
            Write-Host "✅ 다운로드 완료!" -ForegroundColor Green
        }
        "4" { return }
    }
    Read-Host "계속하려면 Enter를 누르세요"
}

# 메인 루프
do {
    Show-Menu
    $choice = Read-Host "명령을 선택하세요 (0-12)"
    
    switch ($choice) {
        "1" { Execute-RemoteCommand "uptime && df -h && free -h" "서버 상태 확인" }
        "2" { Execute-RemoteCommand "nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader,nounits" "GPU 상태 확인" }
        "3" { 
            Write-Host "🚀 서비스 시작 중..." -ForegroundColor Yellow
            ssh -i $KeyPath "$Username@$ServerIP" "cd /home/$Username/HairSwap && nohup python app.py > app.log 2>&1 &"
            Start-Sleep 3
            Execute-RemoteCommand "curl -s http://localhost:8000/health" "서비스 헬스 체크"
        }
        "4" { 
            Write-Host "🛑 서비스 중지 중..." -ForegroundColor Yellow
            ssh -i $KeyPath "$Username@$ServerIP" "pkill -f 'python.*app.py'"
            Write-Host "✅ 서비스가 중지되었습니다." -ForegroundColor Green
            Read-Host "계속하려면 Enter를 누르세요"
        }
        "5" { Show-Logs }
        "6" { 
            Write-Host "🔄 Git 동기화 중..." -ForegroundColor Yellow
            ssh -i $KeyPath "$Username@$ServerIP" "cd /home/$Username/HairSwap && git pull origin dev"
            Write-Host "✅ Git 동기화 완료!" -ForegroundColor Green
            Read-Host "계속하려면 Enter를 누르세요"
        }
        "7" { Execute-RemoteCommand "cd /home/$Username/HairSwap && python test_service.py" "테스트 실행" }
        "8" { Execute-RemoteCommand "ls -la /home/$Username/HairSwap/outputs/" "결과물 확인" }
        "9" { 
            Write-Host "🔧 서버에 접속 중..." -ForegroundColor Yellow
            ssh -i $KeyPath "$Username@$ServerIP"
        }
        "10" { Download-Results }
        "11" { 
            Write-Host "🎯 AI 모델 상태 확인 중..." -ForegroundColor Yellow
            ssh -i $KeyPath "$Username@$ServerIP" "cd /home/$Username/HairSwap && python -c 'import torch; print(f\"CUDA 사용 가능: {torch.cuda.is_available()}\"); print(f\"GPU 개수: {torch.cuda.device_count()}\"); print(f\"현재 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}\")'"
            Read-Host "계속하려면 Enter를 누르세요"
        }
        "12" { Execute-RemoteCommand "ps aux | grep python && echo '---' && free -h && echo '---' && df -h" "메모리 사용량 확인" }
        "0" { 
            Write-Host "👋 종료합니다!" -ForegroundColor Green
            exit
        }
        default { 
            Write-Host "❌ 잘못된 선택입니다." -ForegroundColor Red
            Read-Host "계속하려면 Enter를 누르세요"
        }
    }
} while ($true)
