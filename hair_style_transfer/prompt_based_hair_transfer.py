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

class PromptBasedHairTransfer:
    def __init__(self, device: str = "cpu"):
        """
        프롬프트 기반 헤어스타일 변경 모델 초기화
        Args:
            device: 사용할 디바이스 ("cpu" 또는 "cuda")
        """
        self.device = device
        self.inpaint_pipeline = None
        
        print("프롬프트 기반 헤어스타일 변경 모델 초기화 중...")
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
            hair_top = max(0, top_y - 50)  # 위쪽으로 50픽셀 여유
            hair_mask[:hair_top, :] = 255
            
            return hair_mask
        else:
            # 얼굴 감지 실패시 전체 상단 영역을 헤어로 간주
            hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            hair_mask[:image.shape[0]//3, :] = 255
            return hair_mask
    
    def analyze_hair_style(self, reference_image: np.ndarray) -> dict:
        """
        참조 이미지에서 헤어스타일 특징을 분석합니다.
        Args:
            reference_image: 참조 이미지
        Returns:
            헤어스타일 특징 딕셔너리
        """
        # 헤어 마스크 생성
        hair_mask = self.create_hair_mask(reference_image)
        
        # 헤어 영역 추출
        hair_region = cv2.bitwise_and(reference_image, reference_image, mask=hair_mask)
        
        # 색상 분석
        hair_rgb = cv2.cvtColor(hair_region, cv2.COLOR_BGR2RGB)
        valid_pixels = hair_mask > 0
        
        if np.any(valid_pixels):
            # 평균 색상 계산
            avg_color = np.mean(hair_rgb[valid_pixels], axis=0)
            
            # 색상 기반 헤어 타입 분류
            if avg_color[0] > 150 and avg_color[1] > 150 and avg_color[2] > 150:
                hair_color = "blonde"
            elif avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50:
                hair_color = "black"
            elif avg_color[0] > 100 and avg_color[1] < 100 and avg_color[2] < 100:
                hair_color = "brown"
            elif avg_color[0] > 150 and avg_color[1] > 100 and avg_color[2] < 100:
                hair_color = "red"
            else:
                hair_color = "natural"
            
            # 길이 분석 (마스크의 높이 기반)
            hair_height = np.sum(hair_mask > 0, axis=1)
            max_hair_height = np.max(hair_height)
            image_height = hair_mask.shape[0]
            
            if max_hair_height > image_height * 0.4:
                hair_length = "long"
            elif max_hair_height > image_height * 0.25:
                hair_length = "medium"
            else:
                hair_length = "short"
            
            # 질감 분석 (엣지 검출 기반)
            gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / np.sum(valid_pixels)
            
            if edge_density > 0.1:
                hair_texture = "wavy"
            elif edge_density > 0.05:
                hair_texture = "slightly wavy"
            else:
                hair_texture = "straight"
            
            return {
                "color": hair_color,
                "length": hair_length,
                "texture": hair_texture,
                "avg_color": avg_color
            }
        else:
            return {
                "color": "natural",
                "length": "medium",
                "texture": "straight",
                "avg_color": [128, 128, 128]
            }
    
    def generate_hair_prompt(self, hair_analysis: dict) -> str:
        """
        헤어 분석 결과를 바탕으로 프롬프트를 생성합니다.
        Args:
            hair_analysis: 헤어 분석 결과
        Returns:
            생성된 프롬프트
        """
        color = hair_analysis["color"]
        length = hair_analysis["length"]
        texture = hair_analysis["texture"]
        
        # 기본 프롬프트 구성
        prompt_parts = [
            f"{color} hair",
            f"{length} hair",
            f"{texture} hair",
            "natural looking",
            "high quality",
            "detailed",
            "seamless blending with face"
        ]
        
        # 추가 스타일 정보
        if length == "long":
            prompt_parts.append("flowing hair")
        if texture == "wavy":
            prompt_parts.append("natural waves")
        elif texture == "straight":
            prompt_parts.append("smooth hair")
        
        return ", ".join(prompt_parts)
    
    def transfer_hair_style(
        self, 
        source_image_path: str, 
        reference_image_path: str, 
        output_path: str,
        additional_prompt: str = "",
        negative_prompt: str = "blurry, low quality, distorted, unnatural, obvious seam, bald, no hair",
        num_inference_steps: int = 25,
        guidance_scale: float = 8.0
    ) -> bool:
        """
        참조 이미지의 헤어스타일을 원본 이미지에 적용합니다.
        Args:
            source_image_path: 원본 이미지 경로 (헤어를 바꿀 대상)
            reference_image_path: 참조 이미지 경로 (원하는 헤어스타일)
            output_path: 출력 이미지 경로
            additional_prompt: 추가 프롬프트
            negative_prompt: 부정적 프롬프트
            num_inference_steps: 추론 스텝 수
            guidance_scale: 가이던스 스케일
        Returns:
            성공 여부
        """
        try:
            print("1단계: 이미지 로드 및 헤어스타일 분석 중...")
            # 1. 이미지 로드
            source_image = cv2.imread(source_image_path)
            reference_image = cv2.imread(reference_image_path)
            
            if source_image is None or reference_image is None:
                print("이미지 로드 실패")
                return False
            
            # 2. 참조 이미지에서 헤어스타일 분석
            hair_analysis = self.analyze_hair_style(reference_image)
            print(f"분석된 헤어스타일: {hair_analysis['color']} {hair_analysis['length']} {hair_analysis['texture']} hair")
            
            # 3. 프롬프트 생성
            base_prompt = self.generate_hair_prompt(hair_analysis)
            if additional_prompt:
                final_prompt = f"{base_prompt}, {additional_prompt}"
            else:
                final_prompt = base_prompt
            
            print(f"생성된 프롬프트: {final_prompt}")
            
            print("2단계: 헤어 마스크 생성 중...")
            # 4. 원본 이미지에서 헤어 마스크 생성
            hair_mask = self.create_hair_mask(source_image)
            
            print("3단계: Stable Diffusion으로 헤어스타일 변경 중...")
            # 5. 이미지 크기 조정 (512x512)
            source_resized = cv2.resize(source_image, (512, 512))
            mask_resized = cv2.resize(hair_mask, (512, 512))
            
            # 6. PIL Image로 변환
            source_pil = Image.fromarray(cv2.cvtColor(source_resized, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(mask_resized)
            
            # 7. Stable Diffusion Inpainting 실행
            result = self.inpaint_pipeline(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                image=source_pil,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            # 8. 원본 크기로 리사이즈
            result_np = np.array(result)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            result_resized = cv2.resize(result_bgr, (source_image.shape[1], source_image.shape[0]))
            
            # 9. 결과 저장
            cv2.imwrite(output_path, result_resized)
            print(f"헤어스타일 변경 완료: {output_path}")
            return True
            
        except Exception as e:
            print(f"헤어스타일 변경 중 오류 발생: {e}")
            return False

def main():
    """메인 실행 함수"""
    # 모델 초기화
    hair_transfer = PromptBasedHairTransfer(device="cpu")
    
    # 입력/출력 경로 설정
    source_path = "../images/original/gwangju_long.jpg"  # 헤어를 바꿀 대상
    reference_path = "../images/original/chawoo.jpeg"    # 원하는 헤어스타일 참조
    output_dir = "output"
    output_path = os.path.join(output_dir, "gwangju_with_chawoo_style.png")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"원본 (헤어 변경 대상): {source_path}")
    print(f"참조 (원하는 헤어스타일): {reference_path}")
    print(f"출력: {output_path}")
    
    success = hair_transfer.transfer_hair_style(
        source_image_path=source_path,
        reference_image_path=reference_path,
        output_path=output_path,
        additional_prompt="professional hairstyle, well-groomed"
    )
    
    if success:
        print("헤어스타일 변경이 성공적으로 완료되었습니다!")
    else:
        print("헤어스타일 변경에 실패했습니다.")

if __name__ == "__main__":
    main() 