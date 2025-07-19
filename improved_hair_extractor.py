#!/usr/bin/python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# face_parsing 모듈 경로 추가
sys.path.append(os.path.join('inference', 'face_parsing'))
from model import BiSeNet

class ImprovedHairExtractor:
    def __init__(self):
        """
        개선된 헤어 추출기 초기화
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_classes = 19
        self.net = BiSeNet(n_classes=self.n_classes)
        
        # 모델 로드
        model_path = os.path.join('inference', 'face_parsing', 'res', 'cp', '79999_iter.pth')
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.to(self.device)
        self.net.eval()
        
        print(f"✅ BiSeNet 모델 로드 성공!")
        print(f"   디바이스: {self.device}")
        
        # 이미지 전처리
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    def extract_hair_with_improvements(self, image_path):
        """
        개선된 헤어 추출
        """
        print(f"🔍 개선된 헤어 추출 시작: {image_path}")
        
        # 이미지 로드
        if not os.path.exists(image_path):
            print(f"❌ 이미지 파일이 없습니다: {image_path}")
            return None
        
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        print(f"   원본 크기: {original_size}")
        
        # 512x512로 리사이즈
        image_resized = image.resize((512, 512), Image.BILINEAR)
        image_tensor = self.to_tensor(image_resized)
        image_tensor = torch.unsqueeze(image_tensor, 0).to(self.device)
        
        # 추론
        with torch.no_grad():
            output = self.net(image_tensor)[0]
            parsing = output.squeeze(0).cpu().numpy().argmax(0)
        
        # 1. 기본 헤어 마스크 (클래스 17 - 올바른 매핑)
        basic_hair_mask = (parsing == 17).astype(np.uint8) * 255
        
        # 2. 개선된 헤어 마스크 생성
        improved_hair_mask = self.improve_hair_mask(basic_hair_mask, parsing, image_resized)
        
        # 3. 결과 분석
        basic_pixels = np.sum(basic_hair_mask > 0)
        improved_pixels = np.sum(improved_hair_mask > 0)
        total_pixels = improved_hair_mask.size
        
        print(f"   기본 헤어 픽셀: {basic_pixels}")
        print(f"   개선된 헤어 픽셀: {improved_pixels}")
        print(f"   개선 비율: {(improved_pixels/basic_pixels*100):.1f}%" if basic_pixels > 0 else "   개선 비율: N/A")
        
        # 결과 저장
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 1. 기본 헤어 마스크
        basic_mask_path = f"images/{base_name}_basic_hair_mask.png"
        cv2.imwrite(basic_mask_path, basic_hair_mask)
        print(f"   기본 헤어 마스크: {basic_mask_path}")
        
        # 2. 개선된 헤어 마스크
        improved_mask_path = f"images/{base_name}_improved_hair_mask.png"
        cv2.imwrite(improved_mask_path, improved_hair_mask)
        print(f"   개선된 헤어 마스크: {improved_mask_path}")
        
        # 3. 기본 헤어 영역만 추출 (흰색 배경)
        image_array = np.array(image_resized)
        basic_hair_only = image_array.copy()
        basic_hair_only[basic_hair_mask == 0] = [255, 255, 255]  # 흰색 배경
        
        basic_hair_only_path = f"images/{base_name}_basic_hair_only.jpg"
        cv2.imwrite(basic_hair_only_path, cv2.cvtColor(basic_hair_only, cv2.COLOR_RGB2BGR))
        print(f"   기본 헤어 영역: {basic_hair_only_path}")
        
        # 4. 개선된 헤어 영역만 추출 (흰색 배경)
        improved_hair_only = image_array.copy()
        improved_hair_only[improved_hair_mask == 0] = [255, 255, 255]  # 흰색 배경
        
        improved_hair_only_path = f"images/{base_name}_improved_hair_only.jpg"
        cv2.imwrite(improved_hair_only_path, cv2.cvtColor(improved_hair_only, cv2.COLOR_RGB2BGR))
        print(f"   개선된 헤어 영역: {improved_hair_only_path}")
        
        # 5. 비교 이미지 생성
        comparison = self.create_comparison_image(image_array, basic_hair_mask, improved_hair_mask)
        comparison_path = f"images/{base_name}_comparison.jpg"
        cv2.imwrite(comparison_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"   비교 이미지: {comparison_path}")
        
        return {
            'basic_hair_mask': basic_hair_mask,
            'improved_hair_mask': improved_hair_mask,
            'basic_pixels': basic_pixels,
            'improved_pixels': improved_pixels,
            'original_size': original_size
        }
    
    def improve_hair_mask(self, basic_mask, parsing, image):
        """
        헤어 마스크 개선
        """
        improved_mask = basic_mask.copy()
        
        # 1. 모폴로지 연산으로 노이즈 제거 및 영역 확장
        kernel = np.ones((3,3), np.uint8)
        improved_mask = cv2.morphologyEx(improved_mask, cv2.MORPH_CLOSE, kernel)
        improved_mask = cv2.morphologyEx(improved_mask, cv2.MORPH_OPEN, kernel)
        
        # 2. 피부 영역과의 경계 개선
        skin_mask = (parsing == 1).astype(np.uint8) * 255
        
        # 3. 헤어와 피부 경계에서 헤어 영역 확장
        # 피부 위쪽 영역에서 헤어가 있을 가능성이 높은 부분 추가
        height, width = improved_mask.shape
        top_region = improved_mask[:height//3, :]  # 상단 1/3 영역
        
        # 상단 영역에서 헤어 색상 기반 추가 감지
        image_array = np.array(image)
        hair_color_mask = self.detect_hair_by_color(image_array)
        
        # 상단 영역에서 헤어 색상이 감지되면 헤어 마스크에 추가
        top_hair_color = hair_color_mask[:height//3, :]
        improved_mask[:height//3, :] = np.maximum(improved_mask[:height//3, :], top_hair_color)
        
        # 4. 연결된 컴포넌트 분석으로 작은 노이즈 제거
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(improved_mask, connectivity=8)
        
        # 작은 컴포넌트 제거 (너무 작은 헤어 영역은 노이즈로 간주)
        min_size = 50  # 최소 픽셀 수
        for i in range(1, num_labels):  # 0은 배경이므로 제외
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                improved_mask[labels == i] = 0
        
        return improved_mask
    
    def detect_hair_by_color(self, image):
        """
        색상 기반 헤어 감지
        """
        # HSV 색상 공간으로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 헤어 색상 범위 (갈색, 검정색, 금발 등)
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
        
        hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for lower, upper in hair_colors:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            hair_mask = cv2.bitwise_or(hair_mask, mask)
        
        return hair_mask
    
    def create_comparison_image(self, image, basic_mask, improved_mask):
        """
        비교 이미지 생성 (흰색 배경)
        """
        # 3개 이미지를 가로로 배치
        comparison = np.zeros((image.shape[0], image.shape[1] * 3, 3), dtype=np.uint8)
        
        # 원본 이미지
        comparison[:, :image.shape[1]] = image
        
        # 기본 헤어 마스크 적용 (흰색 배경)
        basic_result = image.copy()
        basic_result[basic_mask == 0] = [255, 255, 255]  # 흰색 배경
        comparison[:, image.shape[1]:image.shape[1]*2] = basic_result
        
        # 개선된 헤어 마스크 적용 (흰색 배경)
        improved_result = image.copy()
        improved_result[improved_mask == 0] = [255, 255, 255]  # 흰색 배경
        comparison[:, image.shape[1]*2:] = improved_result
        
        return comparison

def main():
    """
    메인 함수
    """
    print("=== 개선된 헤어 추출 도구 ===")
    
    # images 폴더의 원본 이미지 파일 찾기
    images_dir = 'images'
    if not os.path.exists(images_dir):
        print(f"❌ {images_dir} 폴더가 없습니다!")
        return
    
    image_files = []
    for file in os.listdir(images_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # 결과 파일들은 제외하고 원본만
            if not any(suffix in file for suffix in ['_hair_mask', '_hair_only', '_parsing_result', '_comparison', '_improved']):
                image_files.append(os.path.join(images_dir, file))
    
    if not image_files:
        print(f"❌ {images_dir} 폴더에 원본 이미지 파일이 없습니다!")
        return
    
    print(f"발견된 원본 이미지 파일들:")
    for i, file in enumerate(image_files, 1):
        print(f"   {i}. {os.path.basename(file)}")
    
    # 개선된 헤어 추출기 초기화
    extractor = ImprovedHairExtractor()
    
    # 각 이미지에서 헤어 추출
    for image_path in image_files:
        print(f"\n{'='*60}")
        result = extractor.extract_hair_with_improvements(image_path)
        
        if result:
            print(f"🎉 헤어 추출 완료!")
            print(f"   기본: {result['basic_pixels']} 픽셀")
            print(f"   개선: {result['improved_pixels']} 픽셀")
        else:
            print("❌ 헤어 추출 실패")
    
    print(f"\n{'='*60}")
    print("✅ 모든 이미지 처리 완료!")
    print("📁 결과 파일들을 확인해보세요:")
    print("   - *_basic_hair_mask.png: 기본 헤어 마스크")
    print("   - *_improved_hair_mask.png: 개선된 헤어 마스크")
    print("   - *_comparison.jpg: 원본/기본/개선 비교 이미지")

if __name__ == "__main__":
    main() 