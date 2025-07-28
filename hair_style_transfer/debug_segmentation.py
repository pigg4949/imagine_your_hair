#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
from typing import Tuple, List

# inference 폴더를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inference', 'face_parsing'))

from model import BiSeNet

class SegmentationDebugger:
    def __init__(self, device: str = "cpu"):
        """
        세그멘테이션 디버깅을 위한 클래스
        """
        self.device = device
        self.face_parsing_model = None
        self._load_model()
    
    def _load_model(self):
        """Face Parsing 모델을 로드합니다."""
        try:
            n_classes = 19
            self.face_parsing_model = BiSeNet(n_classes=n_classes)
            model_path = os.path.join('..', 'inference', 'face_parsing', 'res', 'cp', 'model_final_diss.pth')
            
            if os.path.exists(model_path):
                self.face_parsing_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.face_parsing_model.eval()
                self.face_parsing_model.to(self.device)
                print(f"Face Parsing 모델 로드 완료: {model_path}")
            else:
                print(f"Face Parsing 모델 파일을 찾을 수 없습니다: {model_path}")
                return False
                
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            return False
        
        return True
    
    def get_all_masks(self, image_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        이미지에서 모든 클래스의 마스크를 추출합니다.
        Args:
            image_path: 이미지 경로
        Returns:
            원본 이미지, 모든 클래스 마스크 리스트
        """
        import torchvision.transforms as transforms
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            return None, []
        
        # PIL Image로 변환
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        original_size = pil_image.size
        
        # 이미지 전처리
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # 모델 입력용 크기로 리사이즈
        resized_image = pil_image.resize((512, 512), Image.BILINEAR)
        img_tensor = to_tensor(resized_image).unsqueeze(0).to(self.device)
        
        # 세그멘테이션 실행
        with torch.no_grad():
            out = self.face_parsing_model(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
        # 모든 클래스별 마스크 생성
        masks = []
        for class_id in range(19):  # 19개 클래스
            mask = np.zeros_like(parsing, dtype=np.uint8)
            mask[parsing == class_id] = 255
            
            # 원본 크기로 리사이즈
            mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
            masks.append(mask_resized)
        
        return image, masks
    
    def visualize_masks(self, image: np.ndarray, masks: List[np.ndarray], 
                       target_classes: List[int] = None, save_path: str = None):
        """
        마스크들을 시각화합니다.
        Args:
            image: 원본 이미지
            masks: 마스크 리스트
            target_classes: 시각화할 특정 클래스들
            save_path: 저장 경로
        """
        if target_classes is None:
            target_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        
        # 클래스 이름 정의 (BiSeNet 기준)
        class_names = [
            "background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", 
            "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck", "neck_l", "cloth"
        ]
        
        # 서브플롯 생성
        n_cols = 5
        n_rows = (len(target_classes) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # 원본 이미지 표시
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # 각 클래스별 마스크 표시
        for i, class_id in enumerate(target_classes):
            if class_id < len(masks):
                row = i // n_cols
                col = i % n_cols
                
                # 마스크를 원본 이미지에 오버레이
                mask_overlay = image.copy()
                mask_overlay[masks[class_id] == 255] = [0, 255, 0]  # 녹색으로 표시
                
                axes[row, col].imshow(cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB))
                axes[row, col].set_title(f"Class {class_id}: {class_names[class_id]}")
                axes[row, col].axis('off')
        
        # 빈 서브플롯 숨기기
        for i in range(len(target_classes), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"마스크 시각화 저장: {save_path}")
        
        plt.show()
    
    def find_hair_class(self, image_path: str) -> int:
        """
        헤어 클래스를 찾기 위해 모든 클래스를 테스트합니다.
        Args:
            image_path: 이미지 경로
        Returns:
            가장 가능성 높은 헤어 클래스 인덱스
        """
        image, masks = self.get_all_masks(image_path)
        if image is None:
            return -1
        
        # 헤어로 추정되는 클래스들 (일반적으로 13, 14, 17번이 헤어 관련)
        hair_candidates = [13, 14, 17]
        
        print("헤어 클래스 후보들 분석:")
        for class_id in hair_candidates:
            if class_id < len(masks):
                mask = masks[class_id]
                white_pixels = np.sum(mask == 255)
                total_pixels = mask.shape[0] * mask.shape[1]
                percentage = (white_pixels / total_pixels) * 100
                
                print(f"클래스 {class_id}: {white_pixels} 픽셀 ({percentage:.2f}%)")
        
        # 가장 많은 픽셀을 가진 클래스를 헤어로 선택
        best_class = max(hair_candidates, key=lambda x: np.sum(masks[x] == 255) if x < len(masks) else 0)
        print(f"추천 헤어 클래스: {best_class}")
        
        return best_class

def main():
    """메인 실행 함수"""
    debugger = SegmentationDebugger(device="cpu")
    
    # 테스트할 이미지들
    source_path = "../images/original/gwangju_long.jpg"
    reference_path = "../images/original/chawoo2.jpeg"
    
    print("=== 원본 이미지 헤어 클래스 분석 ===")
    source_hair_class = debugger.find_hair_class(source_path)
    
    print("\n=== 참조 이미지 헤어 클래스 분석 ===")
    reference_hair_class = debugger.find_hair_class(reference_path)
    
    print(f"\n결과:")
    print(f"원본 이미지 헤어 클래스: {source_hair_class}")
    print(f"참조 이미지 헤어 클래스: {reference_hair_class}")
    
    # 마스크 시각화
    print("\n=== 원본 이미지 마스크 시각화 ===")
    source_image, source_masks = debugger.get_all_masks(source_path)
    debugger.visualize_masks(
        source_image, source_masks, 
        target_classes=[13, 14, 17],  # 헤어 관련 클래스들
        save_path="output/source_hair_masks.png"
    )
    
    print("\n=== 참조 이미지 마스크 시각화 ===")
    reference_image, reference_masks = debugger.get_all_masks(reference_path)
    debugger.visualize_masks(
        reference_image, reference_masks,
        target_classes=[13, 14, 17],  # 헤어 관련 클래스들
        save_path="output/reference_hair_masks.png"
    )

if __name__ == "__main__":
    main() 