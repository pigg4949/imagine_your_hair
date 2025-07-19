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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inference', 'face_parsing'))
from model import BiSeNet

class HairSegmenter:
    def __init__(self, model_path=None):
        """
        헤어 세그멘테이션 클래스 초기화
        
        Args:
            model_path: BiSeNet 모델 파일 경로 (None이면 기본 경로 사용)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_classes = 19
        self.net = BiSeNet(n_classes=self.n_classes)
        
        # 모델 로드
        if model_path is None:
            model_path = os.path.join('inference', 'face_parsing', 'res', 'cp', '79999_iter.pth')
        
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ 모델 로드 성공: {model_path}")
        else:
            print(f"⚠️ 모델 파일이 없습니다: {model_path}")
            print("사전 훈련된 모델을 다운로드하거나 훈련해야 합니다.")
        
        self.net.to(self.device)
        self.net.eval()
        
        # 이미지 전처리
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # 클래스별 색상 정의 (19개 클래스) - 올바른 매핑
        self.part_colors = [
            [0, 0, 0],      # 0: background
            [204, 0, 0],    # 1: skin
            [76, 153, 0],   # 2: l_brow (left eyebrow)
            [0, 153, 76],   # 3: r_brow (right eyebrow)
            [153, 0, 76],   # 4: l_eye (left eye)
            [153, 0, 204],  # 5: r_eye (right eye)
            [76, 0, 153],   # 6: eye_g (left eye)
            [0, 204, 153],  # 7: l_ear (left ear)
            [153, 204, 0],  # 8: r_ear (right ear)
            [0, 0, 0],      # 9: ear_r (ear ring)
            [76, 153, 0],   # 10: nose
            [204, 153, 0],  # 11: mouth
            [0, 76, 153],   # 12: u_lip (upper lip)
            [153, 0, 204],  # 13: l_lip (lower lip)
            [0, 255, 255],  # 14: neck
            [0, 255, 0],    # 15: neck_l (necklace)
            [255, 165, 0],  # 16: cloth
            [0, 0, 255],    # 17: hair (파란색) ⭐
            [76, 204, 0]    # 18: hat
        ]
    
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
    
    def segment_face(self, image):
        """
        얼굴 파싱 수행
        
        Args:
            image: 입력 이미지 (PIL Image, numpy array, 또는 파일 경로)
            
        Returns:
            tuple: (parsing_result, original_size)
                - parsing_result: 파싱 결과 (numpy array)
                - original_size: 원본 이미지 크기 (width, height)
        """
        # 이미지 로드 및 전처리
        if isinstance(image, str):
            if os.path.exists(image):
                image = Image.open(image)
            else:
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image}")
        
        original_size = image.size if hasattr(image, 'size') else (image.shape[1], image.shape[0])
        processed_image = self.preprocess_image(image)
        
        # 추론
        with torch.no_grad():
            output = self.net(processed_image)[0]
            parsing = output.squeeze(0).cpu().numpy().argmax(0)
        
        return parsing, original_size
    
    def extract_hair_mask(self, parsing_result):
        """
        헤어 마스크 추출
        
        Args:
            parsing_result: 얼굴 파싱 결과
            
        Returns:
            numpy.ndarray: 헤어 마스크 (0-255)
        """
        # 클래스 17이 헤어 영역 (올바른 매핑)
        hair_mask = (parsing_result == 17).astype(np.uint8) * 255
        return hair_mask
    
    def visualize_parsing(self, image, parsing_result, save_path=None):
        """
        파싱 결과 시각화
        
        Args:
            image: 원본 이미지
            parsing_result: 파싱 결과
            save_path: 저장 경로 (None이면 저장하지 않음)
            
        Returns:
            numpy.ndarray: 시각화된 이미지
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 원본 크기로 리사이즈
        image = image.resize((512, 512), Image.BILINEAR)
        image = np.array(image)
        
        # 파싱 결과를 색상으로 변환 (배경을 흰색으로)
        vis_parsing = parsing_result.copy().astype(np.uint8)
        vis_parsing_color = np.ones((vis_parsing.shape[0], vis_parsing.shape[1], 3)) * 255  # 흰색 배경
        
        for pi in range(1, len(self.part_colors)):
            index = np.where(vis_parsing == pi)
            vis_parsing_color[index[0], index[1], :] = self.part_colors[pi]
        
        vis_parsing_color = vis_parsing_color.astype(np.uint8)
        
        # 원본 이미지와 파싱 결과 합성
        vis_im = cv2.addWeighted(image, 0.4, vis_parsing_color, 0.6, 0)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))
            print(f"시각화 결과 저장: {save_path}")
        
        return vis_im
    
    def segment_hair(self, image, save_results=True):
        """
        헤어 세그멘테이션 메인 함수
        
        Args:
            image: 입력 이미지
            save_results: 결과 저장 여부
            
        Returns:
            dict: 세그멘테이션 결과
        """
        print("🔍 헤어 세그멘테이션 시작...")
        
        # 얼굴 파싱 수행
        parsing_result, original_size = self.segment_face(image)
        
        # 헤어 마스크 추출
        hair_mask = self.extract_hair_mask(parsing_result)
        
        # 결과 저장
        results = {
            'parsing_result': parsing_result,
            'hair_mask': hair_mask,
            'original_size': original_size
        }
        
        if save_results:
            # 파싱 결과 시각화 저장
            vis_path = 'hair_segmentation_result.jpg'
            self.visualize_parsing(image, parsing_result, vis_path)
            
            # 헤어 마스크 저장
            mask_path = 'hair_mask.png'
            cv2.imwrite(mask_path, hair_mask)
            print(f"헤어 마스크 저장: {mask_path}")
        
        print("✅ 헤어 세그멘테이션 완료!")
        return results

def test_hair_segmentation():
    """
    헤어 세그멘테이션 테스트 함수
    """
    print("=== 헤어 세그멘테이션 테스트 ===")
    
    # 이미지 파일 찾기
    image_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print("❌ 이미지 파일을 찾을 수 없습니다!")
        return
    
    # 첫 번째 이미지로 테스트
    image_path = image_files[0]
    print(f"테스트 이미지: {image_path}")
    
    try:
        # 헤어 세그멘터 초기화
        segmenter = HairSegmenter()
        
        # 헤어 세그멘테이션 수행
        results = segmenter.segment_hair(image_path)
        
        print("✅ 테스트 완료!")
        print(f"원본 크기: {results['original_size']}")
        print(f"파싱 결과 형태: {results['parsing_result'].shape}")
        print(f"헤어 마스크 형태: {results['hair_mask'].shape}")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        print("모델 파일이 필요할 수 있습니다.")

if __name__ == "__main__":
    test_hair_segmentation()
