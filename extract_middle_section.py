#!/usr/bin/python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import os

def extract_middle_section(image_path, output_path):
    """
    비교 이미지에서 가운데 섹션만 추출
    
    Args:
        image_path: 비교 이미지 경로
        output_path: 저장할 경로
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return False
    
    height, width = image.shape[:2]
    
    # 가운데 섹션 추출 (전체 너비의 1/3 ~ 2/3)
    start_x = width // 3
    end_x = (width * 2) // 3
    
    middle_section = image[:, start_x:end_x]
    
    # 결과 저장
    cv2.imwrite(output_path, middle_section)
    print(f"✅ 가운데 섹션 추출 완료: {output_path}")
    print(f"   원본 크기: {width}x{height}")
    print(f"   추출 크기: {middle_section.shape[1]}x{middle_section.shape[0]}")
    
    return True

def main():
    """
    메인 함수
    """
    print("=== 가운데 섹션 추출 도구 ===")
    
    # 비교 이미지 파일들 찾기
    comparison_files = []
    for file in os.listdir('images'):
        if file.endswith('_comparison.jpg'):
            comparison_files.append(file)
    
    if not comparison_files:
        print("❌ 비교 이미지 파일을 찾을 수 없습니다!")
        return
    
    print(f"발견된 비교 이미지 파일들:")
    for i, file in enumerate(comparison_files, 1):
        print(f"   {i}. {file}")
    
    # 각 비교 이미지에서 가운데 섹션 추출
    for comparison_file in comparison_files:
        base_name = comparison_file.replace('_comparison.jpg', '')
        output_file = f"images/{base_name}_best_hair_extraction.jpg"
        
        print(f"\n🔍 처리 중: {comparison_file}")
        success = extract_middle_section(f"images/{comparison_file}", output_file)
        
        if success:
            print(f"   ✅ 저장 완료: {output_file}")
        else:
            print(f"   ❌ 처리 실패")
    
    print(f"\n{'='*60}")
    print("✅ 모든 가운데 섹션 추출 완료!")
    print("📁 결과 파일들을 확인해보세요:")
    print("   - *_best_hair_extraction.jpg: 가장 좋은 헤어 추출 결과")

if __name__ == "__main__":
    main() 