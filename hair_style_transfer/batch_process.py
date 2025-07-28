#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from hair_style_transfer import HairStyleTransfer
import argparse

def batch_hair_transfer(
    source_dir: str,
    reference_dir: str,
    output_dir: str,
    reference_hair: str = None,
    prompt: str = "natural hair, high quality, detailed",
    negative_prompt: str = "blurry, low quality, distorted",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5
):
    """
    배치로 헤어스타일 변경을 수행합니다.
    
    Args:
        source_dir: 원본 이미지들이 있는 폴더
        reference_dir: 참조 헤어 이미지들이 있는 폴더
        output_dir: 결과를 저장할 폴더
        reference_hair: 특정 참조 헤어 파일명 (None이면 랜덤 선택)
        prompt: 긍정적 프롬프트
        negative_prompt: 부정적 프롬프트
        num_inference_steps: 추론 스텝 수
        guidance_scale: 가이던스 스케일
    """
    
    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 헤어스타일 변경 모델 초기화
    print("헤어스타일 변경 모델 초기화 중...")
    hair_transfer = HairStyleTransfer(device="cpu")
    print("모델 초기화 완료!")
    
    # 원본 이미지 파일 목록
    source_files = [f for f in os.listdir(source_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 참조 헤어 파일 목록
    reference_files = [f for f in os.listdir(reference_dir) 
                      if f.lower().endswith('.png')]
    
    if not source_files:
        print(f"원본 이미지를 찾을 수 없습니다: {source_dir}")
        return
    
    if not reference_files:
        print(f"참조 헤어 이미지를 찾을 수 없습니다: {reference_dir}")
        return
    
    print(f"처리할 원본 이미지: {len(source_files)}개")
    print(f"사용 가능한 참조 헤어: {len(reference_files)}개")
    
    # 배치 처리
    success_count = 0
    total_count = len(source_files)
    
    for i, source_file in enumerate(source_files, 1):
        print(f"\n[{i}/{total_count}] 처리 중: {source_file}")
        
        # 원본 이미지 경로
        source_path = os.path.join(source_dir, source_file)
        
        # 참조 헤어 선택
        if reference_hair and reference_hair in reference_files:
            ref_file = reference_hair
        else:
            # 랜덤 선택 (첫 번째 파일 사용)
            ref_file = reference_files[0]
        
        reference_path = os.path.join(reference_dir, ref_file)
        
        # 출력 파일명 생성
        name_without_ext = os.path.splitext(source_file)[0]
        ref_name_without_ext = os.path.splitext(ref_file)[0]
        output_filename = f"{name_without_ext}_with_{ref_name_without_ext}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"  원본: {source_file}")
        print(f"  참조 헤어: {ref_file}")
        print(f"  출력: {output_filename}")
        
        try:
            # 헤어스타일 변경 실행
            success = hair_transfer.transfer_hair_style(
                source_image_path=source_path,
                reference_hair_path=reference_path,
                output_path=output_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            if success:
                success_count += 1
                print(f"  ✅ 성공")
            else:
                print(f"  ❌ 실패")
                
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
    
    print(f"\n배치 처리 완료!")
    print(f"성공: {success_count}/{total_count}")
    print(f"결과 저장 위치: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="배치 헤어스타일 변경")
    parser.add_argument("--source", required=True, help="원본 이미지 폴더 경로")
    parser.add_argument("--reference", required=True, help="참조 헤어 폴더 경로")
    parser.add_argument("--output", required=True, help="출력 폴더 경로")
    parser.add_argument("--ref-hair", help="특정 참조 헤어 파일명")
    parser.add_argument("--prompt", default="natural hair, high quality, detailed", 
                       help="긍정적 프롬프트")
    parser.add_argument("--negative-prompt", default="blurry, low quality, distorted",
                       help="부정적 프롬프트")
    parser.add_argument("--steps", type=int, default=20, help="추론 스텝 수")
    parser.add_argument("--guidance", type=float, default=7.5, help="가이던스 스케일")
    
    args = parser.parse_args()
    
    batch_hair_transfer(
        source_dir=args.source,
        reference_dir=args.reference,
        output_dir=args.output,
        reference_hair=args.ref_hair,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance
    )

if __name__ == "__main__":
    main() 