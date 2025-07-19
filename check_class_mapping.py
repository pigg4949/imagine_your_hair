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

def check_class_mapping():
    """
    정확한 클래스 매핑 확인
    """
    print("=== BiSeNet 클래스 매핑 확인 ===")
    
    # face_dataset.py에서 확인한 정확한 클래스 순서
    correct_classes = [
        'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'
    ]
    
    print("정확한 클래스 매핑:")
    for i, class_name in enumerate(correct_classes, 1):
        print(f"   클래스 {i:2d}: {class_name}")
    
    print(f"\n🎯 헤어는 클래스 {correct_classes.index('hair') + 1}입니다!")
    
    # 현재 잘못된 매핑 확인
    print(f"\n❌ 현재 hair_segment.py에서 사용하는 매핑:")
    print(f"   클래스 13을 헤어로 매핑 (잘못됨)")
    print(f"✅ 올바른 매핑:")
    print(f"   클래스 {correct_classes.index('hair') + 1}을 헤어로 매핑")
    
    return correct_classes.index('hair') + 1

def test_correct_hair_extraction():
    """
    올바른 클래스로 헤어 추출 테스트
    """
    print("\n=== 올바른 헤어 추출 테스트 ===")
    
    # 올바른 헤어 클래스 ID
    correct_hair_class = check_class_mapping()
    
    # BiSeNet 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = BiSeNet(n_classes=19)
    
    model_path = os.path.join('inference', 'face_parsing', 'res', 'cp', '79999_iter.pth')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    
    # 이미지 전처리
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # 테스트 이미지
    image_path = 'images/gwangju_long.jpg'
    if not os.path.exists(image_path):
        print(f"❌ 테스트 이미지가 없습니다: {image_path}")
        return
    
    # 이미지 로드 및 처리
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((512, 512), Image.BILINEAR)
    image_tensor = to_tensor(image_resized)
    image_tensor = torch.unsqueeze(image_tensor, 0).to(device)
    
    # 추론
    with torch.no_grad():
        output = net(image_tensor)[0]
        parsing = output.squeeze(0).cpu().numpy().argmax(0)
    
    # 잘못된 클래스 (13)로 헤어 추출
    wrong_hair_mask = (parsing == 13).astype(np.uint8) * 255
    wrong_pixels = np.sum(wrong_hair_mask > 0)
    
    # 올바른 클래스로 헤어 추출
    correct_hair_mask = (parsing == correct_hair_class).astype(np.uint8) * 255
    correct_pixels = np.sum(correct_hair_mask > 0)
    
    print(f"\n📊 헤어 추출 결과 비교:")
    print(f"   잘못된 클래스 13: {wrong_pixels} 픽셀")
    print(f"   올바른 클래스 {correct_hair_class}: {correct_pixels} 픽셀")
    print(f"   개선 비율: {(correct_pixels/wrong_pixels*100):.1f}%" if wrong_pixels > 0 else "   개선 비율: N/A")
    
    # 결과 저장
    image_array = np.array(image_resized)
    
    # 잘못된 헤어 마스크 결과
    wrong_hair_only = image_array.copy()
    wrong_hair_only[wrong_hair_mask == 0] = [0, 0, 0]
    cv2.imwrite('images/wrong_hair_extraction.jpg', cv2.cvtColor(wrong_hair_only, cv2.COLOR_RGB2BGR))
    
    # 올바른 헤어 마스크 결과
    correct_hair_only = image_array.copy()
    correct_hair_only[correct_hair_mask == 0] = [0, 0, 0]
    cv2.imwrite('images/correct_hair_extraction.jpg', cv2.cvtColor(correct_hair_only, cv2.COLOR_RGB2BGR))
    
    print(f"\n📁 결과 파일 저장:")
    print(f"   images/wrong_hair_extraction.jpg: 잘못된 클래스 13 결과")
    print(f"   images/correct_hair_extraction.jpg: 올바른 클래스 {correct_hair_class} 결과")
    
    return correct_hair_class

if __name__ == "__main__":
    correct_hair_class = test_correct_hair_extraction()
    print(f"\n✅ 올바른 헤어 클래스는 {correct_hair_class}입니다!") 