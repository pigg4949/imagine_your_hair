import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
from typing import Tuple, Optional
import requests
from io import BytesIO

# CPU 환경에서 동작하는 최고 성능 모델들
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
import torch.nn.functional as F

class HairStyleTransfer:
    def __init__(self, device: str = "cpu"):
        """
        헤어스타일 변경 모델 초기화
        Args:
            device: 사용할 디바이스 ("cpu" 또는 "cuda")
        """
        self.device = device
        self.inpaint_pipeline = None
        self.face_detector = None
        
        print("헤어스타일 변경 모델 초기화 중...")
        self._load_models()
        print("모델 로드 완료!")
    
    def _load_models(self):
        """필요한 모델들을 로드합니다."""
        try:
            # Inpainting 파이프라인 (CPU 최적화)
            model_id = "runwayml/stable-diffusion-inpainting"
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,  # CPU용 float32
                safety_checker=None,
                requires_safety_checker=False
            )
            self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)
            
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            print("대안 모델을 사용합니다...")
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """대안 모델들을 로드합니다."""
        try:
            # 더 가벼운 모델 사용
            model_id = "CompVis/stable-diffusion-v1-4"
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)
            
        except Exception as e:
            print(f"대안 모델 로드도 실패: {e}")
            raise Exception("모델 로드에 실패했습니다.")
    
    def create_hair_mask(self, image: np.ndarray) -> np.ndarray:
        """
        얼굴에서 헤어 영역만 마스킹합니다.
        Args:
            image: 입력 이미지
        Returns:
            헤어 마스크
        """
        # Mediapipe Face Mesh 사용 (기존 코드 활용)
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
            
            # 헤어 영역 추정 (얼굴 위쪽)
            hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            top_y = min([p[1] for p in face_contour])
            hair_mask[:top_y, :] = 255
            
            return hair_mask
        else:
            # 얼굴 감지 실패시 전체 상단 영역을 헤어로 간주
            hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            hair_mask[:image.shape[0]//3, :] = 255
            return hair_mask
    
    def transfer_hair_style(
        self, 
        source_image_path: str, 
        reference_hair_path: str, 
        output_path: str,
        prompt: str = "natural hair, high quality, detailed",
        negative_prompt: str = "blurry, low quality, distorted",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5
    ) -> bool:
        """
        헤어스타일을 변경합니다.
        Args:
            source_image_path: 원본 이미지 경로
            reference_hair_path: 참조 헤어 이미지 경로
            output_path: 출력 이미지 경로
            prompt: 긍정적 프롬프트
            negative_prompt: 부정적 프롬프트
            num_inference_steps: 추론 스텝 수
            guidance_scale: 가이던스 스케일
        Returns:
            성공 여부
        """
        try:
            # 1. 이미지 로드
            source_image = cv2.imread(source_image_path)
            reference_hair = cv2.imread(reference_hair_path, cv2.IMREAD_UNCHANGED)
            
            if source_image is None or reference_hair is None:
                print("이미지 로드 실패")
                return False
            
            # 2. 헤어 마스크 생성
            hair_mask = self.create_hair_mask(source_image)
            
            # 3. 이미지 크기 통일 (512x512)
            source_resized = cv2.resize(source_image, (512, 512))
            mask_resized = cv2.resize(hair_mask, (512, 512))
            
            # 4. PIL Image로 변환
            source_pil = Image.fromarray(cv2.cvtColor(source_resized, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(mask_resized)
            
            # 5. 참조 헤어 이미지에서 색상 정보 추출하여 프롬프트 개선
            if reference_hair.shape[2] == 4:  # RGBA
                # 투명 배경 제거
                alpha = reference_hair[:, :, 3] / 255.0
                rgb = reference_hair[:, :, :3]
                # 평균 색상 계산
                valid_pixels = alpha > 0.1
                if np.any(valid_pixels):
                    avg_color = np.mean(rgb[valid_pixels], axis=0)
                    # 색상 기반 프롬프트 생성
                    if avg_color[0] > 150 and avg_color[1] > 150 and avg_color[2] > 150:
                        color_desc = "blonde hair"
                    elif avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50:
                        color_desc = "black hair"
                    elif avg_color[0] > 100 and avg_color[1] < 100 and avg_color[2] < 100:
                        color_desc = "brown hair"
                    else:
                        color_desc = "natural hair"
                    
                    enhanced_prompt = f"{color_desc}, {prompt}"
                else:
                    enhanced_prompt = prompt
            else:
                enhanced_prompt = prompt
            
            print(f"사용 프롬프트: {enhanced_prompt}")
            
            # 6. Inpainting 실행
            result = self.inpaint_pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                image=source_pil,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            # 7. 결과 저장
            result.save(output_path)
            print(f"헤어스타일 변경 완료: {output_path}")
            return True
            
        except Exception as e:
            print(f"헤어스타일 변경 중 오류 발생: {e}")
            return False

def main():
    """메인 실행 함수"""
    # 모델 초기화
    hair_transfer = HairStyleTransfer(device="cpu")
    
    # 입력/출력 경로 설정
    source_dir = "../images/original"
    reference_dir = "../images/only_hair"
    output_dir = "output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 예시 실행
    source_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    reference_files = [f for f in os.listdir(reference_dir) if f.lower().endswith('.png')]
    
    if source_files and reference_files:
        source_path = os.path.join(source_dir, source_files[0])
        reference_path = os.path.join(reference_dir, reference_files[0])
        output_path = os.path.join(output_dir, f"transferred_{source_files[0]}")
        
        print(f"원본: {source_path}")
        print(f"참조 헤어: {reference_path}")
        print(f"출력: {output_path}")
        
        success = hair_transfer.transfer_hair_style(
            source_image_path=source_path,
            reference_hair_path=reference_path,
            output_path=output_path
        )
        
        if success:
            print("헤어스타일 변경이 성공적으로 완료되었습니다!")
        else:
            print("헤어스타일 변경에 실패했습니다.")
    else:
        print("입력 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main() 