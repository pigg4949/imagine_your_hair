#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hair Style Transfer Service 테스트 스크립트
"""

import requests
import time
import os
from pathlib import Path

def test_health_check(base_url="http://localhost:8000"):
    """헬스 체크 테스트"""
    print("🔍 헬스 체크 테스트 중...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 헬스 체크 성공: {data}")
            return True
        else:
            print(f"❌ 헬스 체크 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 헬스 체크 오류: {e}")
        return False

def test_examples_endpoint(base_url="http://localhost:8000"):
    """예시 이미지 엔드포인트 테스트"""
    print("🔍 예시 이미지 엔드포인트 테스트 중...")
    try:
        response = requests.get(f"{base_url}/examples", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 예시 이미지 조회 성공: {len(data.get('examples', []))}개 이미지")
            return True
        else:
            print(f"❌ 예시 이미지 조회 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 예시 이미지 조회 오류: {e}")
        return False

def test_upload_image(base_url="http://localhost:8000"):
    """이미지 업로드 테스트"""
    print("🔍 이미지 업로드 테스트 중...")
    
    # 테스트용 이미지 파일 찾기
    test_images = list(Path("images/original").glob("*.jpg")) + list(Path("images/original").glob("*.jpeg")) + list(Path("images/original").glob("*.png"))
    
    if not test_images:
        print("❌ 테스트용 이미지를 찾을 수 없습니다.")
        return False
    
    test_image = test_images[0]
    print(f"📷 테스트 이미지: {test_image}")
    
    try:
        with open(test_image, "rb") as f:
            files = {"file": (test_image.name, f, "image/jpeg")}
            response = requests.post(f"{base_url}/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 이미지 업로드 성공: {data['filename']}")
            return data['filename']
        else:
            print(f"❌ 이미지 업로드 실패: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 이미지 업로드 오류: {e}")
        return None

def test_transfer_hair_style(base_url="http://localhost:8000", source_image=None, reference_image=None):
    """헤어스타일 전송 테스트"""
    print("🔍 헤어스타일 전송 테스트 중...")
    
    if not source_image or not reference_image:
        print("❌ 소스 이미지와 참조 이미지가 필요합니다.")
        return False
    
    try:
        data = {
            "source_image": source_image,
            "reference_image": reference_image,
            "prompt": "natural hair, seamless blending with face, high quality, detailed hairstyle",
            "negative_prompt": "blurry, low quality, distorted, unnatural, obvious seam, bald, no hair, text, watermark, symbols, characters, artifacts"
        }
        
        response = requests.post(f"{base_url}/transfer", data=data, timeout=300)  # 5분 타임아웃
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 헤어스타일 전송 성공: {result['output_filename']}")
            return result['output_filename']
        else:
            print(f"❌ 헤어스타일 전송 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return None
    except Exception as e:
        print(f"❌ 헤어스타일 전송 오류: {e}")
        return None

def main():
    """메인 테스트 함수"""
    print("🧪 Hair Style Transfer Service 테스트 시작")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # 1. 헬스 체크
    if not test_health_check(base_url):
        print("❌ 서비스가 실행되지 않았습니다. 먼저 서비스를 시작해주세요.")
        return
    
    # 2. 예시 이미지 조회
    test_examples_endpoint(base_url)
    
    # 3. 이미지 업로드 테스트
    source_image = test_upload_image(base_url)
    if source_image:
        reference_image = test_upload_image(base_url)
        
        if reference_image:
            # 4. 헤어스타일 전송 테스트
            result_image = test_transfer_hair_style(base_url, source_image, reference_image)
            
            if result_image:
                print(f"🎉 모든 테스트가 성공했습니다!")
                print(f"📁 결과 이미지: {result_image}")
            else:
                print("❌ 헤어스타일 전송 테스트가 실패했습니다.")
        else:
            print("❌ 참조 이미지 업로드가 실패했습니다.")
    else:
        print("❌ 소스 이미지 업로드가 실패했습니다.")
    
    print("=" * 50)
    print("🧪 테스트 완료")

if __name__ == "__main__":
    main()
