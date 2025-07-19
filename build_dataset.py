#!/usr/bin/python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
from pathlib import Path

# face_parsing 모듈 경로 추가
sys.path.append(os.path.join('inference', 'face_parsing'))
from model import BiSeNet

class DatasetBuilder:
    def __init__(self):
        """
        데이터셋 구축기 초기화
        """
        print("🔧 데이터셋 구축기 초기화 중...")
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   디바이스: {self.device}")
        
        # BiSeNet 모델 로드
        self.net = BiSeNet(n_classes=19)
        model_path = os.path.join('inference', 'face_parsing', 'res', 'cp', '79999_iter.pth')
        
        if not os.path.exists(model_path):
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
        
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.to(self.device)
        self.net.eval()
        print("   ✅ BiSeNet 모델 로드 완료")
        
        # 이미지 전처리
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # 출력 폴더 생성
        self.only_hair_dir = Path('images/only_hair')
        self.only_mask_dir = Path('images/only_mask')
        
        self.only_hair_dir.mkdir(exist_ok=True)
        self.only_mask_dir.mkdir(exist_ok=True)
        
        print("   ✅ 출력 폴더 생성 완료")
        print("✅ 데이터셋 구축기 초기화 완료!")
    
    def preprocess_image(self, image):
        """
        이미지 전처리
        
        Args:
            image: PIL Image 또는 numpy array
            
        Returns:
            torch.Tensor: 전처리된 이미지 텐서
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 512x512로 리사이즈
        image = image.resize((512, 512), Image.BILINEAR)
        image = self.to_tensor(image)
        image = torch.unsqueeze(image, 0)
        return image.to(self.device)
    
    def extract_hair_and_mask(self, image_path):
        """
        이미지에서 헤어와 마스크 추출
        
        Args:
            image_path: 입력 이미지 경로
            
        Returns:
            tuple: (hair_image, hair_mask, original_size)
        """
        # 이미지 로드
        if isinstance(image_path, (str, Path)):
            image = Image.open(str(image_path)).convert('RGB')
        else:
            image = image_path
        
        original_size = image.size
        image_resized = image.resize((512, 512), Image.BILINEAR)
        
        # 전처리 및 추론
        image_tensor = self.preprocess_image(image_resized)
        
        with torch.no_grad():
            output = self.net(image_tensor)[0]
            parsing = output.squeeze(0).cpu().numpy().argmax(0)
        
        # 헤어 마스크 추출 (클래스 17)
        hair_mask = (parsing == 17).astype(np.uint8) * 255
        
        # 헤어만 추출한 이미지 (흰색 배경)
        image_array = np.array(image_resized)
        hair_only = image_array.copy()
        hair_only[hair_mask == 0] = [255, 255, 255]  # 흰색 배경
        
        return hair_only, hair_mask, original_size
    
    def build_dataset(self):
        """
        데이터셋 구축 메인 함수
        """
        print("\n=== 데이터셋 구축 시작 ===")
        
        # 원본 이미지 파일들 찾기
        original_dir = Path('images/original')
        if not original_dir.exists():
            print(f"❌ {original_dir} 폴더가 없습니다!")
            return
        
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(original_dir.glob(f'*{ext}'))
            image_files.extend(original_dir.glob(f'*{ext.upper()}'))
        
        # 중복 제거
        image_files = list(set(image_files))
        image_files.sort()  # 정렬
        
        if not image_files:
            print(f"❌ {original_dir} 폴더에 이미지 파일이 없습니다!")
            return
        
        print(f"발견된 이미지 파일들 ({len(image_files)}개):")
        for i, file in enumerate(image_files, 1):
            print(f"   {i:2d}. {file.name}")
        
        # 각 이미지에서 헤어와 마스크 추출
        success_count = 0
        failed_files = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n🔍 처리 중 ({i}/{len(image_files)}): {image_path.name}")
            
            try:
                # 헤어와 마스크 추출
                hair_image, hair_mask, original_size = self.extract_hair_and_mask(image_path)
                
                # 파일명 생성 (확장자 제거)
                base_name = image_path.stem
                
                # 헤어 이미지 저장
                hair_path = self.only_hair_dir / f"{base_name}_hair.jpg"
                cv2.imwrite(str(hair_path), cv2.cvtColor(hair_image, cv2.COLOR_RGB2BGR))
                
                # 마스크 저장
                mask_path = self.only_mask_dir / f"{base_name}_mask.png"
                cv2.imwrite(str(mask_path), hair_mask)
                
                # 헤어 픽셀 수 계산
                hair_pixels = np.sum(hair_mask > 0)
                total_pixels = hair_mask.size
                hair_ratio = (hair_pixels / total_pixels) * 100
                
                print(f"   ✅ 성공: {hair_pixels} 픽셀 ({hair_ratio:.1f}%)")
                print(f"      헤어: {hair_path.name}")
                print(f"      마스크: {mask_path.name}")
                
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ 실패: {str(e)}")
                failed_files.append(image_path.name)
        
        # 결과 요약
        print(f"\n{'='*60}")
        print("📊 데이터셋 구축 결과")
        print(f"   총 이미지: {len(image_files)}개")
        print(f"   성공: {success_count}개")
        print(f"   실패: {len(failed_files)}개")
        
        if failed_files:
            print(f"\n❌ 실패한 파일들:")
            for file in failed_files:
                print(f"   - {file}")
        
        print(f"\n📁 생성된 데이터셋:")
        print(f"   헤어 이미지: {self.only_hair_dir} ({len(list(self.only_hair_dir.glob('*.jpg')))}개)")
        print(f"   마스크: {self.only_mask_dir} ({len(list(self.only_mask_dir.glob('*.png')))}개)")
        
        print(f"\n✅ 데이터셋 구축 완료!")

def main():
    """
    메인 함수
    """
    try:
        builder = DatasetBuilder()
        builder.build_dataset()
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 