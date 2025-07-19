#!/usr/bin/python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def find_image_files():
    """이미지 파일들을 찾는 함수"""
    image_files = []
    
    # 현재 디렉토리와 상위 디렉토리의 images 폴더 확인
    search_paths = [
        '.',  # 현재 디렉토리
        '..',  # 상위 디렉토리
        os.path.join('..', 'images'),  # 상위 디렉토리의 images 폴더
        'images'  # 현재 디렉토리의 images 폴더
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(root, file)
                        if full_path not in image_files:
                            image_files.append(full_path)
    
    return image_files

def simple_hair_detection(image_path):
    """
    간단한 헤어 감지 (모델 없이)
    색상 기반으로 헤어 영역을 대략적으로 추정
    """
    print(f"🔍 이미지 분석: {image_path}")
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    # BGR to RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # HSV 색상 공간으로 변환
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 헤어 색상 범위 정의 (갈색, 검정색, 금발 등)
    hair_colors = [
        # 갈색 헤어
        ([10, 50, 50], [20, 255, 255]),
        # 검정색 헤어
        ([0, 0, 0], [180, 255, 30]),
        # 금발 헤어
        ([20, 100, 100], [30, 255, 255]),
        # 어두운 갈색
        ([0, 50, 20], [20, 255, 100])
    ]
    
    # 헤어 마스크 생성
    hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for lower, upper in hair_colors:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        
        # 색상 범위에 해당하는 픽셀 찾기
        mask = cv2.inRange(image_hsv, lower, upper)
        hair_mask = cv2.bitwise_or(hair_mask, mask)
    
    # 노이즈 제거
    kernel = np.ones((5,5), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
    
    # 결과 시각화
    plt.figure(figsize=(15, 5))
    
    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('원본 이미지')
    plt.axis('off')
    
    # 헤어 마스크
    plt.subplot(1, 3, 2)
    plt.imshow(hair_mask, cmap='gray')
    plt.title('헤어 마스크 (색상 기반)')
    plt.axis('off')
    
    # 마스크 적용된 이미지
    plt.subplot(1, 3, 3)
    masked_image = image_rgb.copy()
    masked_image[hair_mask == 0] = [0, 0, 0]  # 헤어가 아닌 영역을 검정으로
    plt.imshow(masked_image)
    plt.title('헤어 영역 추출')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_hair_detection_result.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    
    return hair_mask

def test_face_parsing_availability():
    """얼굴 파싱 모델 사용 가능성 테스트"""
    print("=== 얼굴 파싱 모델 테스트 ===")
    
    try:
        # face_parsing 모듈 import 시도
        import sys
        sys.path.append(os.path.join('..', 'inference', 'face_parsing'))
        
        from model import BiSeNet
        print("✅ BiSeNet 모델 클래스 import 성공")
        
        # 모델 파일 확인
        model_path = os.path.join('..', 'inference', 'face_parsing', 'res', 'cp', '79999_iter.pth')
        if os.path.exists(model_path):
            print(f"✅ 모델 파일 존재: {model_path}")
            return True
        else:
            print(f"⚠️ 모델 파일 없음: {model_path}")
            print("사전 훈련된 모델을 다운로드해야 합니다.")
            return False
            
    except ImportError as e:
        print(f"❌ 모듈 import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("=== 헤어 세그멘테이션 테스트 ===")
    
    # 1. 이미지 파일 찾기
    image_files = find_image_files()
    if not image_files:
        print("❌ 이미지 파일을 찾을 수 없습니다!")
        return
    
    print(f"발견된 이미지 파일들: {image_files}")
    
    # 2. 얼굴 파싱 모델 사용 가능성 확인
    parsing_available = test_face_parsing_availability()
    
    # 3. 간단한 헤어 감지 테스트
    print("\n=== 간단한 헤어 감지 테스트 ===")
    for image_path in image_files[:2]:  # 처음 2개 이미지만 테스트
        hair_mask = simple_hair_detection(image_path)
        if hair_mask is not None:
            print(f"✅ {image_path} 헤어 감지 완료")
            print(f"   헤어 마스크 형태: {hair_mask.shape}")
            print(f"   헤어 픽셀 수: {np.sum(hair_mask > 0)}")
    
    print("\n=== 테스트 완료 ===")
    print("결과 이미지: simple_hair_detection_result.jpg")
    
    if not parsing_available:
        print("\n💡 다음 단계:")
        print("1. BiSeNet 사전 훈련 모델 다운로드")
        print("2. 정확한 얼굴 파싱 구현")
        print("3. 헤어스타일 변경 알고리즘 개발")

if __name__ == "__main__":
    main() 