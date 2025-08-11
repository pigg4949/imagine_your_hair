#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import subprocess
import time

def test_data_preparation():
    """데이터 전처리 프로세스를 테스트합니다."""
    
    print("🧪 데이터 전처리 프로세스 테스트 시작")
    print("="*60)
    
    # 1. 테스트 환경 확인
    print("1️⃣ 테스트 환경 확인 중...")
    
    # 입력 디렉토리 확인
    input_dir = Path("images/original")
    if not input_dir.exists():
        print(f"❌ 입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        return False
    
    # 이미지 파일 개수 확인
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))
    print(f"✅ 입력 이미지 {len(image_files)}개 발견: {input_dir}")
    
    if len(image_files) == 0:
        print("❌ 테스트할 이미지가 없습니다.")
        return False
    
    # 2. 기본 데이터 전처리 테스트
    print("\n2️⃣ 기본 데이터 전처리 테스트...")
    
    output_dir = "test_training_data"
    
    # 기존 테스트 디렉토리 정리
    if Path(output_dir).exists():
        import shutil
        shutil.rmtree(output_dir)
        print(f"🧹 기존 테스트 디렉토리 정리: {output_dir}")
    
    # 향상된 데이터 전처리 실행
    cmd = [
        sys.executable, "training/enhanced_data_preparation.py",
        "--input_dir", str(input_dir),
        "--output_dir", output_dir,
        "--target_size", "512", "512",
        "--min_face_size", "80",
        "--quality_threshold", "0.6",
        "--enhance_quality",
        "--create_preview"
    ]
    
    print(f"실행 명령: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        processing_time = time.time() - start_time
        
        print("✅ 데이터 전처리 성공!")
        print(f"⏱️ 처리 시간: {processing_time:.2f}초")
        
        # 출력 확인
        if result.stdout:
            print("\n📋 처리 로그:")
            print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 데이터 전처리 실패: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    
    # 3. 결과 확인
    print("\n3️⃣ 결과 확인...")
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"❌ 출력 디렉토리가 생성되지 않았습니다: {output_dir}")
        return False
    
    # 처리된 이미지 개수 확인
    processed_images = list(output_path.glob("*.jpg"))
    print(f"✅ 처리된 이미지: {len(processed_images)}개")
    
    # 통계 파일 확인
    stats_file = output_path / "processing_stats.json"
    if stats_file.exists():
        print(f"✅ 통계 파일 생성됨: {stats_file}")
        
        # 통계 내용 출력
        import json
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        print("\n📊 처리 통계:")
        print(f"  - 총 이미지: {stats['total_images']}")
        print(f"  - 처리된 이미지: {stats['processed_images']}")
        print(f"  - 거부된 이미지: {stats['rejected_images']}")
        print(f"  - 처리 시간: {stats['processing_time']:.2f}초")
        
        if stats['quality_scores']:
            import numpy as np
            print(f"  - 평균 품질 점수: {np.mean(stats['quality_scores']):.3f}")
    
    # 미리보기 파일 확인
    preview_file = output_path / "preview.png"
    if preview_file.exists():
        print(f"✅ 미리보기 생성됨: {preview_file}")
    
    # 검증 디렉토리 확인
    validation_dir = output_path / "validation"
    if validation_dir.exists():
        validation_images = list(validation_dir.glob("*.jpg"))
        print(f"✅ 검증 데이터: {len(validation_images)}개")
    
    # 4. 품질 검사
    print("\n4️⃣ 품질 검사...")
    
    if len(processed_images) > 0:
        print("✅ 데이터 전처리가 성공적으로 완료되었습니다!")
        
        # 첫 번째 이미지 정보 출력
        first_image = processed_images[0]
        print(f"📸 샘플 이미지: {first_image.name}")
        
        # 이미지 크기 확인
        import cv2
        img = cv2.imread(str(first_image))
        if img is not None:
            height, width = img.shape[:2]
            print(f"  - 크기: {width}x{height}")
        
        return True
    else:
        print("❌ 처리된 이미지가 없습니다.")
        return False

def test_dreambooth_requirements():
    """DreamBooth 학습 요구사항을 확인합니다."""
    
    print("\n🔍 DreamBooth 학습 요구사항 확인...")
    
    # 필요한 패키지 확인
    required_packages = [
        "torch", "diffusers", "transformers", "accelerate", 
        "datasets", "safetensors", "xformers"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (설치 필요)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 설치 필요한 패키지: {', '.join(missing_packages)}")
        print("다음 명령으로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ 모든 필요한 패키지가 설치되어 있습니다.")
        return True

def main():
    """메인 테스트 함수"""
    
    print("🚀 AI 헤어스왑 데이터 전처리 테스트")
    print("="*60)
    
    # 1. 데이터 전처리 테스트
    prep_success = test_data_preparation()
    
    # 2. DreamBooth 요구사항 확인
    req_success = test_dreambooth_requirements()
    
    # 3. 결과 요약
    print("\n" + "="*60)
    print("📋 테스트 결과 요약")
    print("="*60)
    
    if prep_success and req_success:
        print("🎉 모든 테스트가 성공했습니다!")
        print("\n다음 단계:")
        print("1. 학습 데이터 확인: test_training_data/")
        print("2. DreamBooth 학습 시작:")
        print("   python training/hair_style_dreambooth.py \\")
        print("     --training_data_dir test_training_data \\")
        print("     --output_dir trained_models \\")
        print("     --hair_style_token hst \\")
        print("     --max_train_steps 1000 \\")
        print("     --use_lora")
        return True
    else:
        print("❌ 일부 테스트가 실패했습니다.")
        if not prep_success:
            print("- 데이터 전처리 문제 확인 필요")
        if not req_success:
            print("- 필요한 패키지 설치 필요")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
