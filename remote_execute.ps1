# 원격 서버 명령 실행 스크립트 (PowerShell)
# 사용법: .\remote_execute.ps1 -ServerIP "34.123.45.67" -Username "your-username" -Command "nvidia-smi"

param(
    [Parameter(Mandatory=$true)]
    [string]$ServerIP,
    
    [Parameter(Mandatory=$true)]
    [string]$Username,
    
    [Parameter(Mandatory=$true)]
    [string]$Command,
    
    [Parameter(Mandatory=$false)]
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_rsa",
    
    [Parameter(Mandatory=$false)]
    [switch]$Background = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Interactive = $false
)

Write-Host "🚀 원격 명령 실행: $Command" -ForegroundColor Green
Write-Host "📍 서버: $Username@$ServerIP" -ForegroundColor Cyan

# SSH 키 확인
if (-not (Test-Path $KeyPath)) {
    Write-Host "❌ SSH 키를 찾을 수 없습니다: $KeyPath" -ForegroundColor Red
    exit 1
}

try {
    if ($Background) {
        # 백그라운드 실행
        Write-Host "🔄 백그라운드에서 실행 중..." -ForegroundColor Yellow
        Start-Job -ScriptBlock {
            param($KeyPath, $Username, $ServerIP, $Command)
            ssh -i $KeyPath "$Username@$ServerIP" "$Command"
        } -ArgumentList $KeyPath, $Username, $ServerIP, $Command
        
        Write-Host "✅ 백그라운드 작업이 시작되었습니다." -ForegroundColor Green
        Write-Host "📊 작업 상태 확인: Get-Job" -ForegroundColor Gray
        
    } elseif ($Interactive) {
        # 대화형 실행
        Write-Host "💬 대화형 모드로 연결 중..." -ForegroundColor Yellow
        ssh -i $KeyPath "$Username@$ServerIP"
        
    } else {
        # 일반 실행
        Write-Host "⚡ 명령 실행 중..." -ForegroundColor Yellow
        $result = ssh -i $KeyPath "$Username@$ServerIP" "$Command"
        Write-Host "📋 실행 결과:" -ForegroundColor Cyan
        Write-Host $result -ForegroundColor White
    }
    
} catch {
    Write-Host "❌ 명령 실행 실패: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "✅ 명령 실행 완료!" -ForegroundColor Green
