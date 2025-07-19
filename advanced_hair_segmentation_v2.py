#!/usr/bin/python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import torch
from PIL import Image
import os
import sys
from pathlib import Path
import mediapipe as mp
from transformers import AutoImageProcessor, AutoModelForImageSegmentation
import requests
from io import BytesIO

# face_parsing 모듈 경로 추가
sys.path.append(os.path.join('inference', 'face_parsing'))
from model import BiSeNet

class AdvancedHairSegmentationV2:
    def __init__(self):
        """
        고급 헤어 세그멘테이션 시스템 V2 초기화
        """
        print("🔧 고급 헤어 세그멘테이션 시스템 V2 초기화 중...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   디바이스: {self.device}")
        
        # 1. BiSeNet (기존)
        self.bisenet = self._init_bisenet()
        
        # 2. MediaPipe Face Mesh
        self.face_mesh = self._init_face_mesh()
        
        # 3. OneFormer (새로운 고성능 모델)
        self.oneformer = self._init_oneformer()
        
        # 4. SAM (Segment Anything Model)
        self.sam = self._init_sam()
        
        # 출력 폴더
        self.output_dir = Path('images/advanced_hair_masks')
        self.output_dir.mkdir(exist_ok=True)
        
        print("   ✅ 모든 모델 초기화 완료")
        print("✅ 고급 헤어 세그멘테이션 시스템 V2 초기화 완료!")
    
    def _init_bisenet(self):
        """
        BiSeNet 초기화
        """
        try:
            net = BiSeNet(n_classes=19)
            model_path = os.path.join('inference', 'face_parsing', 'res', 'cp', '79999_iter.pth')
            
            if os.path.exists(model_path):
                net.load_state_dict(torch.load(model_path, map_location=self.device))
                net.to(self.device)
                net.eval()
                print("   ✅ BiSeNet 로드 완료")
                return net
            else:
                print("   ⚠️ BiSeNet 모델 파일 없음")
                return None
        except Exception as e:
            print(f"   ⚠️ BiSeNet 초기화 실패: {e}")
            return None
    
    def _init_face_mesh(self):
        """
        MediaPipe Face Mesh 초기화
        """
        try:
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            print("   ✅ MediaPipe Face Mesh 로드 완료")
            return face_mesh
        except Exception as e:
            print(f"   ⚠️ MediaPipe Face Mesh 초기화 실패: {e}")
            return None
    
    def _init_oneformer(self):
        """
        OneFormer 초기화 (고성능 세그멘테이션)
        """
        try:
            # OneFormer 모델 로드
            model_name = "shi-labs/oneformer_ade20k_swin_tiny_patch4_window7_224"
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageSegmentation.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            
            print("   ✅ OneFormer 로드 완료")
            return {"model": model, "processor": processor}
        except Exception as e:
            print(f"   ⚠️ OneFormer 초기화 실패: {e}")
            return None
    
    def _init_sam(self):
        """
        SAM (Segment Anything Model) 초기화
        """
        try:
            # SAM 모델 로드
            model_name = "facebook/sam-vit-base"
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageSegmentation.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            
            print("   ✅ SAM 로드 완료")
            return {"model": model, "processor": processor}
        except Exception as e:
            print(f"   ⚠️ SAM 초기화 실패: {e}")
            return None
    
    def extract_perfect_hair_mask_v2(self, image_path):
        """
        V2 완벽한 헤어 마스크 추출 (4중 모델 결합)
        """
        print(f"🎯 V2 완벽한 헤어 마스크 추출 중: {Path(image_path).name}")
        
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # 1. BiSeNet 기반 세그멘테이션
        bisenet_mask = self._bisenet_segmentation_v2(image_array)
        
        # 2. OneFormer 기반 세그멘테이션
        oneformer_mask = self._oneformer_segmentation(image_array)
        
        # 3. SAM 기반 세그멘테이션
        sam_mask = self._sam_segmentation(image_array)
        
        # 4. 얼굴 구조 기반 마스킹
        structural_mask = self._structural_mask_v2(image_array)
        
        # 5. 색상 기반 세그멘테이션 (개선된 버전)
        color_mask = self._color_based_segmentation_v2(image_array)
        
        # 6. 모든 마스크 결합 및 개선
        combined_mask = self._combine_masks_v2(
            bisenet_mask, oneformer_mask, sam_mask, structural_mask, color_mask
        )
        final_mask = self._improve_mask_v2(combined_mask, image_array)
        
        print(f"   ✅ V2 완벽한 헤어 마스크 추출 완료")
        return final_mask
    
    def _bisenet_segmentation_v2(self, image_array):
        """
        BiSeNet 기반 세그멘테이션 (개선된 버전)
        """
        if self.bisenet is None:
            return np.zeros(image_array.shape[:2], dtype=np.uint8)
        
        try:
            # 이미지 전처리
            image_pil = Image.fromarray(image_array)
            image_resized = image_pil.resize((512, 512), Image.BILINEAR)
            
            # 정규화
            image_tensor = torch.from_numpy(np.array(image_resized)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            image_tensor = (image_tensor - 0.485) / 0.229
            image_tensor = image_tensor.to(self.device)
            
            # 추론
            with torch.no_grad():
                output = self.bisenet(image_tensor)[0]
                parsing = output.squeeze(0).cpu().numpy().argmax(0)
            
            # 헤어 클래스 (17) 마스크
            hair_mask = (parsing == 17).astype(np.uint8) * 255
            
            # 원본 크기로 리사이즈
            hair_mask = cv2.resize(hair_mask, (image_array.shape[1], image_array.shape[0]))
            
            return hair_mask
            
        except Exception as e:
            print(f"   ⚠️ BiSeNet 세그멘테이션 실패: {e}")
            return np.zeros(image_array.shape[:2], dtype=np.uint8)
    
    def _oneformer_segmentation(self, image_array):
        """
        OneFormer 기반 세그멘테이션
        """
        if self.oneformer is None:
            return np.zeros(image_array.shape[:2], dtype=np.uint8)
        
        try:
            # 이미지 전처리
            image_pil = Image.fromarray(image_array)
            
            # OneFormer 추론
            inputs = self.oneformer["processor"](images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.oneformer["model"](**inputs)
                pred_masks = outputs.pred_masks.squeeze(0)
                pred_labels = outputs.pred_labels.squeeze(0)
            
            # 헤어 관련 클래스 찾기 (person, hair 등)
            hair_mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
            
            for i, label in enumerate(pred_labels):
                label_id = label.item()
                # person, hair, head 등 관련 클래스
                if label_id in [0, 1, 2, 3]:  # 일반적인 사람 관련 클래스
                    mask = pred_masks[i].cpu().numpy()
                    mask = cv2.resize(mask, (image_array.shape[1], image_array.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    hair_mask = cv2.bitwise_or(hair_mask, mask)
            
            return hair_mask
            
        except Exception as e:
            print(f"   ⚠️ OneFormer 세그멘테이션 실패: {e}")
            return np.zeros(image_array.shape[:2], dtype=np.uint8)
    
    def _sam_segmentation(self, image_array):
        """
        SAM 기반 세그멘테이션
        """
        if self.sam is None:
            return np.zeros(image_array.shape[:2], dtype=np.uint8)
        
        try:
            # 이미지 전처리
            image_pil = Image.fromarray(image_array)
            
            # SAM 추론
            inputs = self.sam["processor"](images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.sam["model"](**inputs)
                pred_masks = outputs.pred_masks.squeeze(0)
            
            # 가장 큰 마스크 선택 (헤어 영역일 가능성)
            hair_mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
            
            for i in range(pred_masks.shape[0]):
                mask = pred_masks[i].cpu().numpy()
                mask = cv2.resize(mask, (image_array.shape[1], image_array.shape[0]))
                mask = (mask > 0.5).astype(np.uint8) * 255
                
                # 마스크 크기 확인
                mask_area = np.sum(mask > 0)
                if mask_area > 1000:  # 충분히 큰 마스크만
                    hair_mask = cv2.bitwise_or(hair_mask, mask)
            
            return hair_mask
            
        except Exception as e:
            print(f"   ⚠️ SAM 세그멘테이션 실패: {e}")
            return np.zeros(image_array.shape[:2], dtype=np.uint8)
    
    def _structural_mask_v2(self, image):
        """
        얼굴 구조 기반 마스킹 (개선된 버전)
        """
        if self.face_mesh is None:
            return np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        try:
            results = self.face_mesh.process(image)
            
            if not results.multi_face_landmarks:
                return np.ones(image.shape[:2], dtype=np.uint8) * 255
            
            landmarks = results.multi_face_landmarks[0]
            
            # 확장된 헤어 관련 랜드마크 인덱스들
            hair_landmarks = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
                151, 337, 299, 333, 285, 252, 390, 357, 455, 324, 362, 289, 398, 366, 380, 379, 401, 378,
                153, 149, 177, 150, 151, 137, 173, 59, 133, 94, 235, 128, 163, 22, 55, 104, 68, 110
            ]
            
            # 랜드마크를 픽셀 좌표로 변환
            points = []
            for idx in hair_landmarks:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    points.append([x, y])
            
            if len(points) < 3:
                return np.ones(image.shape[:2], dtype=np.uint8) * 255
            
            points = np.array(points)
            
            # 헤어 영역의 바운딩 박스 계산
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            
            # 더 큰 마진으로 확장
            margin = 50
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(image.shape[1], x_max + margin)
            y_max = min(image.shape[0], y_max + margin)
            
            # 마스크 생성
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[y_min:y_max, x_min:x_max] = 255
            
            return mask
            
        except Exception as e:
            print(f"   ⚠️ 구조적 마스킹 실패: {e}")
            return np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    def _color_based_segmentation_v2(self, image):
        """
        색상 기반 세그멘테이션 (개선된 버전)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 더 정교한 헤어 색상 범위
        hair_colors = [
            ([0, 0, 0], [180, 255, 50]),      # 검정
            ([10, 50, 50], [20, 255, 255]),   # 갈색
            ([20, 100, 100], [30, 255, 255]), # 금발
            ([0, 50, 20], [20, 255, 100]),    # 어두운 갈색
            ([0, 0, 50], [180, 30, 150]),     # 회색
            ([0, 0, 30], [180, 50, 100]),     # 매우 어두운 색
            ([0, 20, 20], [20, 255, 200]),    # 다양한 갈색 톤
            ([0, 0, 20], [180, 255, 80])      # 어두운 톤
        ]
        
        hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for lower, upper in hair_colors:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            hair_mask = cv2.bitwise_or(hair_mask, mask)
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
        
        return hair_mask
    
    def _combine_masks_v2(self, bisenet_mask, oneformer_mask, sam_mask, structural_mask, color_mask):
        """
        V2 마스크 결합 (가중 평균 + 지능형 결합)
        """
        # 가중치 설정
        weights = {
            'bisenet': 0.25,
            'oneformer': 0.25,
            'sam': 0.20,
            'structural': 0.15,
            'color': 0.15
        }
        
        # 마스크 정규화
        masks = {
            'bisenet': bisenet_mask.astype(np.float32) / 255.0,
            'oneformer': oneformer_mask.astype(np.float32) / 255.0,
            'sam': sam_mask.astype(np.float32) / 255.0,
            'structural': structural_mask.astype(np.float32) / 255.0,
            'color': color_mask.astype(np.float32) / 255.0
        }
        
        # 가중 평균 계산
        combined = np.zeros_like(masks['bisenet'])
        for name, weight in weights.items():
            combined += masks[name] * weight
        
        # 이진화
        combined = (combined > 0.3).astype(np.uint8) * 255
        
        return combined
    
    def _improve_mask_v2(self, mask, image):
        """
        V2 마스크 개선 (고급 후처리)
        """
        # 1. 모폴로지 연산 (더 정교한)
        kernel_small = np.ones((5, 5), np.uint8)
        kernel_large = np.ones((9, 9), np.uint8)
        
        # 열기 연산으로 노이즈 제거
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # 닫기 연산으로 구멍 메우기
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        
        # 2. 연결 요소 분석으로 작은 노이즈 제거
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        min_size = 1000  # 더 큰 최소 크기
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                mask[labels == i] = 0
        
        # 3. 경계 부드럽게 (더 정교한)
        mask = cv2.GaussianBlur(mask, (15, 15), 2)
        
        # 4. 이진화
        mask = (mask > 127).astype(np.uint8) * 255
        
        # 5. 최종 모폴로지 연산
        kernel_final = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_final)
        
        return mask
    
    def process_all_images(self):
        """
        origin 폴더의 모든 이미지에서 헤어 마스크 추출
        """
        print("\n=== 모든 이미지에서 헤어 마스크 추출 ===")
        
        # 원본 이미지 폴더
        original_dir = Path('images/original')
        original_images = list(original_dir.glob('*.jpg')) + list(original_dir.glob('*.jpeg')) + list(original_dir.glob('*.png'))
        
        if not original_images:
            print("❌ 원본 이미지가 없습니다!")
            return
        
        print(f"📁 처리할 이미지: {len(original_images)}개")
        
        for i, image_path in enumerate(original_images, 1):
            print(f"\n[{i}/{len(original_images)}] 처리 중: {image_path.name}")
            
            try:
                # V2 헤어 마스크 추출
                hair_mask = self.extract_perfect_hair_mask_v2(str(image_path))
                
                # 마스크 저장
                mask_name = f"{image_path.stem}_hair_mask_v2.png"
                mask_path = self.output_dir / mask_name
                cv2.imwrite(str(mask_path), hair_mask)
                
                print(f"   ✅ 마스크 저장: {mask_name}")
                
            except Exception as e:
                print(f"   ❌ 처리 실패: {e}")
        
        print(f"\n🎉 모든 이미지 처리 완료!")
        print(f"📁 저장 위치: {self.output_dir}")

def main():
    """
    메인 함수
    """
    try:
        segmenter = AdvancedHairSegmentationV2()
        
        print("\n=== 고급 헤어 세그멘테이션 시스템 V2 ===")
        print("1. 모든 이미지에서 헤어 마스크 추출")
        print("q. 종료")
        
        while True:
            choice = input("\n모드를 선택하세요 (1, q): ").strip().lower()
            
            if choice == 'q':
                print("👋 프로그램을 종료합니다.")
                break
            elif choice == '1':
                segmenter.process_all_images()
            else:
                print("❌ 1 또는 q를 입력하세요.")
            
            print(f"\n{'='*60}")
    
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 