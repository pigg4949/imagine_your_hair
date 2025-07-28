#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
from typing import Tuple, Optional
from diffusers import StableDiffusionInpaintPipeline
import torch.nn.functional as F

class SimpleHairTransfer:
    def __init__(self, device: str = "cpu"):
        """
        간단한 헤어스타일 변경 모델 초기화
        Args:
            device: 사용할 디바이스 ("cpu" 또는 "cuda")
        """
        self.device = device
        self.inpaint_pipeline = None
        
        print("간단한 헤어스타일 변경 모델 초기화 중...")
        self._load_models()
        print("모델 로드 완료!")
    
    def _load_models(self):
        """필요한 모델들을 로드합니다."""
        try:
            # Inpainting 파이프라인 (CPU 최적화)
            model_id = "runwayml/stable-diffusion-inpainting"
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)
            
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            raise Exception("모델 로드에 실패했습니다.")
    
    def create_hair_mask(self, image: np.ndarray) -> np.ndarray:
        """
        얼굴에서 헤어 영역만 마스킹합니다.
        Args:
            image: 입력 이미지
        Returns:
            헤어 마스크
        """
        import mediapipe as mp
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # 얼굴 윤곽선 추출
            face_contour = []
            for i in range(0, 27):
                x = int(landmarks.landmark[i].x * image.shape[1])
                y = int(landmarks.landmark[i].y * image.shape[0])
                face_contour.append([x, y])
            
            # 헤어 영역 추정 (얼굴 위쪽 + 약간의 여유)
            hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            top_y = min([p[1] for p in face_contour])
            # 헤어 영역을 조금 더 넓게 설정
            hair_top = max(0, top_y - 30)  # 위쪽으로 30픽셀 여유
            hair_mask[:hair_top, :] = 255
            
            return hair_mask
        else:
            # 얼굴 감지 실패시 전체 상단 영역을 헤어로 간주
            hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            hair_mask[:image.shape[0]//3, :] = 255
            return hair_mask
    
    def transfer_hair_style(
        self, 
        source_image_path: str, 
        output_path: str,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, unnatural, obvious seam, bald, no hair, text, watermark, symbols, characters",
        num_inference_steps: int = 30,
        guidance_scale: float = 8.5
    ) -> bool:
        """
        사용자 프롬프트로 헤어스타일을 변경합니다.
        Args:
            source_image_path: 원본 이미지 경로 (헤어를 바꿀 대상)
            output_path: 출력 이미지 경로
            prompt: 헤어스타일 변경 프롬프트
            negative_prompt: 부정적 프롬프트
            num_inference_steps: 추론 스텝 수
            guidance_scale: 가이던스 스케일
        Returns:
            성공 여부
        """
        try:
            print("1단계: 원본 이미지 로드 중...")
            # 1. 원본 이미지 로드
            source_image = cv2.imread(source_image_path)
            
            if source_image is None:
                print("이미지 로드 실패")
                return False
            
            print("2단계: 헤어 마스크 생성 중...")
            # 2. 헤어 마스크 생성
            hair_mask = self.create_hair_mask(source_image)
            
            print("3단계: Stable Diffusion으로 헤어스타일 변경 중...")
            # 3. 이미지 크기 조정 (512x512)
            source_resized = cv2.resize(source_image, (512, 512))
            mask_resized = cv2.resize(hair_mask, (512, 512))
            
            # 4. PIL Image로 변환
            source_pil = Image.fromarray(cv2.cvtColor(source_resized, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(mask_resized)
            
            print(f"사용 프롬프트: {prompt}")
            print(f"부정적 프롬프트: {negative_prompt}")
            
            # 5. Stable Diffusion Inpainting 실행
            result = self.inpaint_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=source_pil,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            # 6. 원본 크기로 리사이즈
            result_np = np.array(result)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            result_resized = cv2.resize(result_bgr, (source_image.shape[1], source_image.shape[0]))
            
            # 7. 결과 저장
            cv2.imwrite(output_path, result_resized)
            print(f"헤어스타일 변경 완료: {output_path}")
            return True
            
        except Exception as e:
            print(f"헤어스타일 변경 중 오류 발생: {e}")
            return False

def main():
    """메인 실행 함수"""
    # 모델 초기화
    hair_transfer = SimpleHairTransfer(device="cpu")
    
    # 입력/출력 경로 설정
    source_path = "../images/original/gwangju_long.jpg"
    output_dir = "output"
    output_path = os.path.join(output_dir, "gwangju_simple_hair_change.png")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"원본: {source_path}")
    print(f"출력: {output_path}")
    
    # 사용자가 직접 입력하는 프롬프트 (참조 이미지 기반)
    prompt = "change the hair style to match the reference image: black hair, medium length, slightly wavy, natural looking, professional hairstyle, well-groomed, seamless blending with face"
    
    success = hair_transfer.transfer_hair_style(
        source_image_path=source_path,
        output_path=output_path,
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted, unnatural, obvious seam, bald, no hair, text, watermark, symbols, characters, artifacts",
        num_inference_steps=30,
        guidance_scale=8.5
    )
    
    if success:
        print("헤어스타일 변경이 성공적으로 완료되었습니다!")
    else:
        print("헤어스타일 변경에 실패했습니다.")

if __name__ == "__main__":
    main() 