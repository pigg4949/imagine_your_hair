#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from hair_style_transfer import HairStyleTransfer
import os

def test_specific_hair_transfer():
    """특정 이미지들로 헤어스타일 변경을 테스트합니다."""
    
    # 모델 초기화
    print("헤어스타일 변경 모델 초기화 중...")
    hair_transfer = HairStyleTransfer(device="cpu")
    print("모델 초기화 완료!")
    
    # 입력/출력 경로 설정
    source_path = "../images/original/gwangju_long.jpg"
    reference_path = "../images/only_hair/chawoo.png"
    output_dir = "output"
    output_path = os.path.join(output_dir, "gwangju_with_chawoo_hair.png")
    
    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"원본 이미지: {source_path}")
    print(f"참조 헤어: {reference_path}")
    print(f"출력 경로: {output_path}")
    
    # 파일 존재 확인
    if not os.path.exists(source_path):
        print(f"❌ 원본 이미지를 찾을 수 없습니다: {source_path}")
        return
    
    if not os.path.exists(reference_path):
        print(f"❌ 참조 헤어 이미지를 찾을 수 없습니다: {reference_path}")
        return
    
    print("\n헤어스타일 변경 시작...")
    
    # 헤어스타일 변경 실행
    success = hair_transfer.transfer_hair_style(
        source_image_path=source_path,
        reference_hair_path=reference_path,
        output_path=output_path,
        prompt="natural black hair, medium length, slightly wavy, high quality, detailed",
        negative_prompt="blurry, low quality, distorted, unnatural",
        num_inference_steps=25,
        guidance_scale=8.0
    )
    
    if success:
        print(f"\n✅ 헤어스타일 변경 성공!")
        print(f"결과 이미지: {output_path}")
    else:
        print(f"\n❌ 헤어스타일 변경 실패!")

if __name__ == "__main__":
    test_specific_hair_transfer() 