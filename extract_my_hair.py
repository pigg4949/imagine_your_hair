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

class MyHairExtractor:
    def __init__(self):
        """
        개인 헤어 추출기 초기화
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
    
    def extract_hair_from_image(self, image_path):
        """
        이미지에서 헤어 추출
        """
        print(f"🔍 헤어 추출 시작: {image_path}")
        
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
        
        # 헤어 마스크 추출 (클래스 13)
        hair_mask = (parsing == 13).astype(np.uint8) * 255
        
        # 결과 분석
        hair_pixels = np.sum(hair_mask > 0)
        total_pixels = hair_mask.size
        hair_percentage = (hair_pixels / total_pixels) * 100
        
        print(f"   헤어 픽셀 수: {hair_pixels}")
        print(f"   헤어 비율: {hair_percentage:.2f}%")
        
        # 결과 저장
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 1. 헤어 마스크 저장
        mask_path = f"images/{base_name}_hair_mask.png"
        cv2.imwrite(mask_path, hair_mask)
        print(f"   헤어 마스크 저장: {mask_path}")
        
        # 2. 헤어 영역만 추출한 이미지 저장
        image_array = np.array(image_resized)
        hair_only = image_array.copy()
        hair_only[hair_mask == 0] = [0, 0, 0]  # 헤어가 아닌 영역을 검정으로
        
        hair_only_path = f"images/{base_name}_hair_only.jpg"
        cv2.imwrite(hair_only_path, cv2.cvtColor(hair_only, cv2.COLOR_RGB2BGR))
        print(f"   헤어 영역만 추출: {hair_only_path}")
        
        # 3. 파싱 결과 시각화 저장
        # 색상 매핑 (실제 이미지에서 보이는 색상 기준)
        part_colors = [
            [0, 0, 0],      # 0: background
            [204, 0, 0],    # 1: skin (빨간색)
            [255, 255, 0],  # 2: nose (노란색)
            [128, 0, 128],  # 3: eye_g (보라색)
            [128, 0, 128],  # 4: l_eye (보라색)
            [128, 0, 128],  # 5: r_eye (보라색)
            [0, 255, 0],    # 6: l_brow (연두색)
            [0, 255, 0],    # 7: r_brow (연두색)
            [0, 255, 0],    # 8: l_ear (연두색)
            [0, 255, 0],    # 9: r_ear (연두색)
            [255, 192, 203], # 10: mouth (분홍색)
            [255, 192, 203], # 11: u_lip (분홍색)
            [255, 192, 203], # 12: l_lip (분홍색)
            [0, 0, 255],    # 13: hair (파란색) ⭐
            [0, 255, 255],  # 14: hat (밝은 초록색)
            [0, 255, 0],    # 15: earring (연두색)
            [255, 165, 0],  # 16: necklace (주황색)
            [255, 165, 0],  # 17: cloth (주황색)
            [128, 0, 128]   # 18: glasses (보라색)
        ]
        
        # 파싱 결과를 색상으로 변환
        vis_parsing = parsing.copy().astype(np.uint8)
        vis_parsing_color = np.zeros((vis_parsing.shape[0], vis_parsing.shape[1], 3)) + 255
        
        for pi in range(1, len(part_colors)):
            index = np.where(vis_parsing == pi)
            vis_parsing_color[index[0], index[1], :] = part_colors[pi]
        
        vis_parsing_color = vis_parsing_color.astype(np.uint8)
        
        # 원본 이미지와 파싱 결과 합성
        vis_im = cv2.addWeighted(image_array, 0.4, vis_parsing_color, 0.6, 0)
        
        parsing_path = f"images/{base_name}_parsing_result.jpg"
        cv2.imwrite(parsing_path, cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))
        print(f"   파싱 결과 시각화: {parsing_path}")
        
        return {
            'hair_mask': hair_mask,
            'hair_pixels': hair_pixels,
            'hair_percentage': hair_percentage,
            'original_size': original_size
        }

def main():
    """
    메인 함수
    """
    print("=== 개인 헤어 추출 도구 ===")
    
    # images 폴더의 모든 이미지 파일 찾기
    images_dir = 'images'
    if not os.path.exists(images_dir):
        print(f"❌ {images_dir} 폴더가 없습니다!")
        return
    
    image_files = []
    for file in os.listdir(images_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # 기존 결과 파일들은 제외
            if not any(suffix in file for suffix in ['_hair_mask', '_hair_only', '_parsing_result']):
                image_files.append(os.path.join(images_dir, file))
    
    if not image_files:
        print(f"❌ {images_dir} 폴더에 이미지 파일이 없습니다!")
        print("   지원 형식: .jpg, .jpeg, .png")
        return
    
    print(f"발견된 이미지 파일들:")
    for i, file in enumerate(image_files, 1):
        print(f"   {i}. {os.path.basename(file)}")
    
    # 헤어 추출기 초기화
    extractor = MyHairExtractor()
    
    # 각 이미지에서 헤어 추출
    for image_path in image_files:
        print(f"\n{'='*60}")
        result = extractor.extract_hair_from_image(image_path)
        
        if result:
            if result['hair_pixels'] > 0:
                print(f"🎉 헤어 추출 성공! ({result['hair_pixels']} 픽셀, {result['hair_percentage']:.2f}%)")
            else:
                print("⚠️ 헤어 영역이 감지되지 않았습니다.")
        else:
            print("❌ 헤어 추출 실패")
    
    print(f"\n{'='*60}")
    print("✅ 모든 이미지 처리 완료!")
    print("📁 결과 파일들은 images 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 