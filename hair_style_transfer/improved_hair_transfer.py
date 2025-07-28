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

# inference 폴더를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inference', 'face_parsing'))

from model import BiSeNet

class ImprovedHairTransfer:
    def __init__(self, device: str = "cpu"):
        """
        개선된 헤어스타일 변경 모델 초기화
        Args:
            device: 사용할 디바이스 ("cpu" 또는 "cuda")
        """
        self.device = device
        self.inpaint_pipeline = None
        self.face_parsing_model = None
        
        print("개선된 헤어스타일 변경 모델 초기화 중...")
        self._load_models()
        print("모델 로드 완료!")
    
    def _load_models(self):
        """필요한 모델들을 로드합니다."""
        try:
            # 1. Stable Diffusion Inpainting 파이프라인
            model_id = "runwayml/stable-diffusion-inpainting"
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.inpaint_pipeline = self.inpaint_pipeline.to(self.device)
            
            # 2. Face Parsing 모델 (BiSeNet)
            n_classes = 19
            self.face_parsing_model = BiSeNet(n_classes=n_classes)
            self.face_parsing_model.to(self.device)
            
            # 사전학습된 모델 로드
            model_path = os.path.join('..', 'inference', 'face_parsing', 'res', 'cp', 'model_final_diss.pth')
            if os.path.exists(model_path):
                self.face_parsing_model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Face Parsing 모델 로드 완료: {model_path}")
            else:
                print(f"Face Parsing 모델 파일을 찾을 수 없습니다: {model_path}")
                raise Exception("Face Parsing 모델이 필요합니다.")
            
            self.face_parsing_model.eval()
            
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            raise Exception("모델 로드에 실패했습니다.")
    
    def get_combined_hair_mask(self, image: np.ndarray) -> np.ndarray:
        """
        여러 헤어 클래스를 조합하여 더 정확한 헤어 마스크를 생성합니다.
        Args:
            image: 입력 이미지
        Returns:
            조합된 헤어 마스크
        """
        import torchvision.transforms as transforms
        
        # 이미지 전처리
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # PIL Image로 변환 및 리사이즈
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        original_size = pil_image.size
        resized_image = pil_image.resize((512, 512), Image.BILINEAR)
        
        # 텐서 변환
        img_tensor = to_tensor(resized_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.face_parsing_model(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
        # 헤어 관련 클래스들 조합
        hair_classes = [13, 14, 17]  # hair, hat, neck_l (헤어 관련 클래스들)
        
        # 조합된 헤어 마스크 생성
        combined_mask = np.zeros_like(parsing, dtype=np.uint8)
        for class_id in hair_classes:
            combined_mask[parsing == class_id] = 255
        
        # 모폴로지 연산으로 마스크 정제
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 원본 크기로 리사이즈
        combined_mask = cv2.resize(combined_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        return combined_mask
    
    def get_face_mask(self, image: np.ndarray) -> np.ndarray:
        """
        얼굴 영역 마스크를 생성합니다.
        Args:
            image: 입력 이미지
        Returns:
            얼굴 마스크
        """
        import torchvision.transforms as transforms
        
        # 이미지 전처리
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # PIL Image로 변환 및 리사이즈
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        original_size = pil_image.size
        resized_image = pil_image.resize((512, 512), Image.BILINEAR)
        
        # 텐서 변환
        img_tensor = to_tensor(resized_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.face_parsing_model(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
        # 얼굴 클래스들 (피부, 눈, 코, 입 등)
        face_classes = [1, 2, 3, 4, 5, 6, 10, 11, 12]  # skin, nose, eye_g, l_eye, r_eye, l_brow, r_brow, mouth, u_lip, l_lip
        
        # 얼굴 마스크 생성
        face_mask = np.zeros_like(parsing, dtype=np.uint8)
        for class_id in face_classes:
            face_mask[parsing == class_id] = 255
        
        # 원본 크기로 리사이즈
        face_mask = cv2.resize(face_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        return face_mask
    
    def extract_hair_with_style(self, image: np.ndarray, hair_mask: np.ndarray) -> np.ndarray:
        """
        헤어 마스크를 사용하여 이미지에서 헤어를 추출합니다.
        Args:
            image: 원본 이미지
            hair_mask: 헤어 마스크
        Returns:
            헤어만 추출된 이미지 (투명 배경)
        """
        # RGBA 이미지 생성
        rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = image  # RGB 채널
        rgba_image[:, :, 3] = hair_mask  # 알파 채널
        
        return rgba_image
    
    def align_hair_to_face(self, hair_image: np.ndarray, face_image: np.ndarray, 
                          hair_mask: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
        """
        헤어를 얼굴에 맞게 정렬하고 크기를 조정합니다.
        Args:
            hair_image: 헤어 이미지 (RGBA)
            face_image: 얼굴 이미지
            hair_mask: 헤어 마스크
            face_mask: 얼굴 마스크
        Returns:
            정렬된 헤어 이미지
        """
        # 얼굴 영역의 바운딩 박스 찾기
        face_contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not face_contours:
            return hair_image
        
        # 가장 큰 얼굴 영역 선택
        face_contour = max(face_contours, key=cv2.contourArea)
        face_bbox = cv2.boundingRect(face_contour)
        
        # 헤어 영역의 바운딩 박스 찾기
        hair_contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not hair_contours:
            return hair_image
        
        # 가장 큰 헤어 영역 선택
        hair_contour = max(hair_contours, key=cv2.contourArea)
        hair_bbox = cv2.boundingRect(hair_contour)
        
        # 헤어를 얼굴 크기에 맞게 스케일링
        face_width = face_bbox[2]
        face_height = face_bbox[3]
        hair_width = hair_bbox[2]
        hair_height = hair_bbox[3]
        
        # 스케일 계산 (헤어가 얼굴보다 약간 크게)
        scale_x = face_width / hair_width * 1.2
        scale_y = face_height / hair_height * 1.2
        scale = min(scale_x, scale_y)
        
        # 헤어 이미지 리사이즈
        new_width = int(hair_image.shape[1] * scale)
        new_height = int(hair_image.shape[0] * scale)
        resized_hair = cv2.resize(hair_image, (new_width, new_height))
        
        # 얼굴 중심에 헤어 배치
        face_center_x = face_bbox[0] + face_bbox[2] // 2
        face_center_y = face_bbox[1] + face_bbox[3] // 2
        
        hair_center_x = new_width // 2
        hair_center_y = new_height // 2
        
        # 새로운 이미지 생성
        result = np.zeros((face_image.shape[0], face_image.shape[1], 4), dtype=np.uint8)
        
        # 헤어를 얼굴 중심에 배치
        start_x = face_center_x - hair_center_x
        start_y = face_center_y - hair_center_y
        
        # 경계 확인
        if start_x < 0:
            resized_hair = resized_hair[:, -start_x:]
            start_x = 0
        if start_y < 0:
            resized_hair = resized_hair[-start_y:, :]
            start_y = 0
        
        end_x = min(start_x + resized_hair.shape[1], result.shape[1])
        end_y = min(start_y + resized_hair.shape[0], result.shape[0])
        
        if start_x < end_x and start_y < end_y:
            result[start_y:end_y, start_x:end_x] = resized_hair[:end_y-start_y, :end_x-start_x]
        
        return result
    
    def composite_hair_on_face(self, face_image: np.ndarray, hair_image: np.ndarray, 
                             hair_mask: np.ndarray) -> np.ndarray:
        """
        얼굴 이미지에 헤어를 자연스럽게 합성합니다.
        Args:
            face_image: 얼굴 이미지
            hair_image: 헤어 이미지 (RGBA)
            hair_mask: 헤어 마스크
        Returns:
            합성된 이미지
        """
        # RGBA 처리
        if hair_image.shape[2] == 4:
            alpha = hair_image[:, :, 3] / 255.0
            rgb = hair_image[:, :, :3]
            
            # 헤어 마스크와 알파 채널 결합
            mask_3ch = np.stack([hair_mask, hair_mask, hair_mask], axis=2) / 255.0
            combined_mask = mask_3ch * alpha[:, :, np.newaxis]
            
            # 합성
            result = face_image * (1 - combined_mask) + rgb * combined_mask
        else:
            # RGB만 있는 경우
            mask_3ch = np.stack([hair_mask, hair_mask, hair_mask], axis=2) / 255.0
            result = face_image * (1 - mask_3ch) + hair_image * mask_3ch
        
        return result.astype(np.uint8)
    
    def refine_with_stable_diffusion(self, composite_image: np.ndarray, hair_mask: np.ndarray,
                                   prompt: str, negative_prompt: str) -> np.ndarray:
        """
        Stable Diffusion으로 합성된 이미지를 자연스럽게 정제합니다.
        Args:
            composite_image: 합성된 이미지
            hair_mask: 헤어 마스크
            prompt: 긍정적 프롬프트
            negative_prompt: 부정적 프롬프트
        Returns:
            정제된 이미지
        """
        # 이미지 크기 조정 (512x512)
        composite_resized = cv2.resize(composite_image, (512, 512))
        mask_resized = cv2.resize(hair_mask, (512, 512))
        
        # PIL Image로 변환
        composite_pil = Image.fromarray(cv2.cvtColor(composite_resized, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask_resized)
        
        # Stable Diffusion Inpainting으로 정제
        result = self.inpaint_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=composite_pil,
            mask_image=mask_pil,
            num_inference_steps=25,
            guidance_scale=8.0
        ).images[0]
        
        # 원본 크기로 리사이즈
        result_np = np.array(result)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        result_resized = cv2.resize(result_bgr, (composite_image.shape[1], composite_image.shape[0]))
        
        return result_resized
    
    def transfer_hair_style(
        self, 
        source_image_path: str, 
        reference_image_path: str, 
        output_path: str,
        prompt: str = "natural hair, seamless blending with face, high quality, detailed hairstyle",
        negative_prompt: str = "blurry, low quality, distorted, unnatural, obvious seam, bald, no hair, text, watermark, symbols, characters, artifacts"
    ) -> bool:
        """
        개선된 방식으로 헤어스타일을 변경합니다.
        Args:
            source_image_path: 원본 이미지 경로
            reference_image_path: 참조 이미지 경로
            output_path: 출력 이미지 경로
            prompt: 긍정적 프롬프트
            negative_prompt: 부정적 프롬프트
        Returns:
            성공 여부
        """
        try:
            print("1단계: 이미지 로드 및 세그멘테이션 중...")
            # 1. 이미지 로드
            source_image = cv2.imread(source_image_path)
            reference_image = cv2.imread(reference_image_path)
            
            if source_image is None or reference_image is None:
                print("이미지 로드 실패")
                return False
            
            # 2. 원본 이미지 세그멘테이션
            source_hair_mask = self.get_combined_hair_mask(source_image)
            source_face_mask = self.get_face_mask(source_image)
            print("원본 이미지 세그멘테이션 완료")
            
            # 3. 참조 이미지 세그멘테이션
            reference_hair_mask = self.get_combined_hair_mask(reference_image)
            reference_face_mask = self.get_face_mask(reference_image)
            print("참조 이미지 세그멘테이션 완료")
            
            print("2단계: 헤어 추출 및 정렬 중...")
            # 4. 참조 이미지에서 헤어 추출
            reference_hair = self.extract_hair_with_style(reference_image, reference_hair_mask)
            
            # 5. 헤어를 얼굴에 맞게 정렬
            aligned_hair = self.align_hair_to_face(
                reference_hair, source_image, reference_hair_mask, source_face_mask
            )
            
            print("3단계: 헤어 합성 중...")
            # 6. 원본 얼굴에 참조 헤어 합성
            composite_image = self.composite_hair_on_face(
                source_image, aligned_hair, source_hair_mask
            )
            
            print("4단계: Stable Diffusion으로 자연스럽게 연결 중...")
            # 7. Stable Diffusion으로 정제
            refined_image = self.refine_with_stable_diffusion(
                composite_image, source_hair_mask, prompt, negative_prompt
            )
            
            # 8. 결과 저장
            cv2.imwrite(output_path, refined_image)
            print(f"헤어스타일 변경 완료: {output_path}")
            return True
            
        except Exception as e:
            print(f"헤어스타일 변경 중 오류 발생: {e}")
            return False

def main():
    """메인 실행 함수"""
    # 모델 초기화
    hair_transfer = ImprovedHairTransfer(device="cpu")
    
    # 입력/출력 경로 설정
    source_path = "../images/original/gwangju_long.jpg"
    reference_path = "../images/original/chawoo2.jpeg"
    output_dir = "output"
    output_path = os.path.join(output_dir, "gwangju_improved_hair_change.png")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"원본 (헤어 변경 대상): {source_path}")
    print(f"참조 (원하는 헤어스타일): {reference_path}")
    print(f"출력: {output_path}")
    
    success = hair_transfer.transfer_hair_style(
        source_image_path=source_path,
        reference_image_path=reference_path,
        output_path=output_path,
        prompt="natural hair, seamless blending with face, high quality, detailed hairstyle, professional look",
        negative_prompt="blurry, low quality, distorted, unnatural, obvious seam, bald, no hair, text, watermark, symbols, characters, artifacts"
    )
    
    if success:
        print("개선된 헤어스타일 변경이 성공적으로 완료되었습니다!")
    else:
        print("헤어스타일 변경에 실패했습니다.")

if __name__ == "__main__":
    main() 