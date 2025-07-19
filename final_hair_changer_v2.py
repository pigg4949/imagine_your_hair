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

# face_parsing 모듈 경로 추가
sys.path.append(os.path.join('inference', 'face_parsing'))
from model import BiSeNet

class FinalHairChangerV2:
    def __init__(self):
        """
        최종 헤어스타일 변경기 V2 초기화
        """
        print("🔧 최종 헤어스타일 변경기 V2 초기화 중...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   디바이스: {self.device}")
        
        # BiSeNet 모델 로드
        self.bisenet = self._init_bisenet()
        
        # MediaPipe Face Mesh 초기화
        self.face_mesh = self._init_face_mesh()
        
        # 출력 폴더
        self.output_dir = Path('images/final_results_v2')
        self.output_dir.mkdir(exist_ok=True)
        
        # V2 마스크 폴더
        self.v2_masks_dir = Path('images/advanced_hair_masks')
        
        print("   ✅ 모델 초기화 완료")
        print("✅ 최종 헤어스타일 변경기 V2 초기화 완료!")
    
    def _init_bisenet(self):
        """
        BiSeNet 초기화
        """
        net = BiSeNet(n_classes=19)
        model_path = os.path.join('inference', 'face_parsing', 'res', 'cp', '79999_iter.pth')
        
        if os.path.exists(model_path):
            net.load_state_dict(torch.load(model_path, map_location=self.device))
            net.to(self.device)
            net.eval()
            print("   ✅ BiSeNet 로드 완료")
        else:
            print("   ⚠️ BiSeNet 모델 파일 없음")
        
        return net
    
    def _init_face_mesh(self):
        """
        MediaPipe Face Mesh 초기화
        """
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("   ✅ MediaPipe Face Mesh 로드 완료")
        return face_mesh
    
    def load_v2_hair_mask(self, image_name):
        """
        V2 헤어 마스크 로드
        """
        # 이미지 이름에서 확장자 제거
        base_name = Path(image_name).stem
        
        # V2 마스크 파일 경로
        mask_path = self.v2_masks_dir / f"{base_name}_hair_mask_v2.png"
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            print(f"   ✅ V2 마스크 로드: {mask_path.name}")
            return mask
        else:
            print(f"   ⚠️ V2 마스크 없음: {mask_path.name}")
            return None
    
    def extract_perfect_hair_mask_v2(self, image_path):
        """
        V2 완벽한 헤어 마스크 추출 (기존 + V2 마스크 결합)
        """
        print(f"🎯 V2 완벽한 헤어 마스크 추출 중: {Path(image_path).name}")
        
        # 1. V2 마스크 로드 시도
        v2_mask = self.load_v2_hair_mask(Path(image_path).name)
        
        if v2_mask is not None:
            print(f"   ✅ V2 마스크 사용")
            return v2_mask
        
        # 2. V2 마스크가 없으면 기존 방식 사용
        print(f"   🔄 기존 방식으로 마스크 생성")
        
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # BiSeNet 기반 세그멘테이션
        bisenet_mask = self._bisenet_segmentation(image_array)
        
        # 색상 기반 세그멘테이션
        color_mask = self._color_based_segmentation(image_array)
        
        # 얼굴 구조 기반 마스킹
        structural_mask = self._structural_mask(image_array)
        
        # 마스크 결합 및 개선
        combined_mask = self._combine_masks(bisenet_mask, color_mask, structural_mask)
        final_mask = self._improve_mask(combined_mask, image_array)
        
        print(f"   ✅ 기존 방식 마스크 생성 완료")
        return final_mask
    
    def _bisenet_segmentation(self, image_array):
        """
        BiSeNet 기반 세그멘테이션
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
    
    def _color_based_segmentation(self, image):
        """
        색상 기반 세그멘테이션
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 다양한 헤어 색상 범위
        hair_colors = [
            ([0, 0, 0], [180, 255, 50]),      # 검정
            ([10, 50, 50], [20, 255, 255]),   # 갈색
            ([20, 100, 100], [30, 255, 255]), # 금발
            ([0, 50, 20], [20, 255, 100]),    # 어두운 갈색
            ([0, 0, 50], [180, 30, 150])      # 회색
        ]
        
        hair_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for lower, upper in hair_colors:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            hair_mask = cv2.bitwise_or(hair_mask, mask)
        
        return hair_mask
    
    def _structural_mask(self, image):
        """
        얼굴 구조 기반 마스킹
        """
        results = self.face_mesh.process(image)
        
        if not results.multi_face_landmarks:
            return np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        landmarks = results.multi_face_landmarks[0]
        
        # 헤어 관련 랜드마크 인덱스들
        hair_landmarks = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # 랜드마크를 픽셀 좌표로 변환
        points = []
        for idx in hair_landmarks:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            points.append([x, y])
        
        points = np.array(points)
        
        # 헤어 영역의 바운딩 박스 계산
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        
        # 약간 확장
        margin = 30
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        y_max = min(image.shape[0], y_max + margin)
        
        # 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y_min:y_max, x_min:x_max] = 255
        
        return mask
    
    def _combine_masks(self, bisenet_mask, color_mask, structural_mask):
        """
        마스크 결합
        """
        # 가중 평균으로 결합
        combined = (bisenet_mask * 0.5 + color_mask * 0.3 + structural_mask * 0.2).astype(np.uint8)
        
        # 이진화
        combined = (combined > 127).astype(np.uint8) * 255
        
        return combined
    
    def _improve_mask(self, mask, image):
        """
        마스크 개선
        """
        # 모폴로지 연산
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 경계 부드럽게
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        
        # 작은 노이즈 제거
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        min_size = 500
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                mask[labels == i] = 0
        
        return mask
    
    def apply_hairstyle_change_v2(self, source_image_path, target_hair_path):
        """
        V2 헤어스타일 변경
        """
        print(f"🎭 V2 헤어스타일 변경 시작...")
        print(f"   원본: {Path(source_image_path).name}")
        print(f"   타겟: {Path(target_hair_path).name}")
        
        try:
            # 1. V2 완벽한 헤어 마스크 추출
            hair_mask = self.extract_perfect_hair_mask_v2(source_image_path)
            
            # 2. 원본 이미지 로드
            source_image = cv2.imread(source_image_path)
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            
            # 3. 얼굴 구조 분석
            face_info = self.analyze_face_structure(source_image)
            
            # 4. 타겟 헤어 로드 및 전처리
            target_hair = cv2.imread(target_hair_path)
            target_hair = cv2.cvtColor(target_hair, cv2.COLOR_BGR2RGB)
            target_hair = cv2.resize(target_hair, (source_image.shape[1], source_image.shape[0]))
            
            # 5. 얼굴 방향에 맞게 조정
            if face_info['orientation']:
                target_hair = self._adjust_to_face_orientation(
                    target_hair, face_info['orientation']
                )
            
            # 6. 고급 블렌딩
            result_image = self._advanced_blending_v2(
                source_image, target_hair, hair_mask, face_info
            )
            
            # 7. 결과 저장
            output_path = self._save_result_v2(source_image_path, target_hair_path, result_image)
            
            print(f"   ✅ V2 헤어스타일 변경 완료: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"   ❌ 오류 발생: {str(e)}")
            return None
    
    def analyze_face_structure(self, image):
        """
        얼굴 구조 분석
        """
        results = self.face_mesh.process(image)
        
        face_info = {
            'landmarks': None,
            'orientation': None
        }
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            face_info['landmarks'] = landmarks
            
            # 얼굴 방향 계산
            left_ear = landmarks.landmark[234]
            right_ear = landmarks.landmark[454]
            face_angle = np.arctan2(right_ear.x - left_ear.x, 0.1) * 180 / np.pi
            face_info['orientation'] = face_angle
            
            print(f"   ✅ 얼굴 방향: {face_angle:.1f}도")
        
        return face_info
    
    def _adjust_to_face_orientation(self, hair_image, angle):
        """
        얼굴 방향에 맞게 헤어 조정
        """
        if abs(angle) < 5:
            return hair_image
        
        center = (hair_image.shape[1] // 2, hair_image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_hair = cv2.warpAffine(hair_image, rotation_matrix, (hair_image.shape[1], hair_image.shape[0]))
        
        return rotated_hair
    
    def _advanced_blending_v2(self, source_image, hair_image, hair_mask, face_info):
        """
        V2 고급 블렌딩 (개선된 버전)
        """
        # 마스크 정규화
        mask_norm = hair_mask.astype(np.float32) / 255.0
        
        # 원본에서 헤어 영역 완전 제거
        skin_color = self._estimate_skin_color_v2(source_image, hair_mask)
        cleaned_source = source_image.copy()
        
        for c in range(3):
            cleaned_source[:, :, c] = (
                source_image[:, :, c] * (1 - mask_norm) +
                skin_color[c] * mask_norm
            ).astype(np.uint8)
        
        # 헤어 합성
        blended = cleaned_source.copy()
        for c in range(3):
            blended[:, :, c] = (
                cleaned_source[:, :, c] * (1 - mask_norm) +
                hair_image[:, :, c] * mask_norm
            ).astype(np.uint8)
        
        # V2 경계 부드럽게 처리 (더 정교한)
        kernel = np.ones((21, 21), np.float32) / 441  # 더 큰 커널
        smooth_mask = cv2.filter2D(mask_norm, -1, kernel)
        
        # 다단계 블렌딩
        for c in range(3):
            blended[:, :, c] = (
                cleaned_source[:, :, c] * (1 - smooth_mask) +
                blended[:, :, c] * smooth_mask
            ).astype(np.uint8)
        
        # 최종 색상 보정
        blended = self._color_correction_v2(blended, source_image, hair_mask)
        
        return blended.astype(np.uint8)
    
    def _estimate_skin_color_v2(self, image, hair_mask):
        """
        V2 피부색 추정 (개선된 버전)
        """
        kernel = np.ones((21, 21), np.uint8)  # 더 큰 커널
        hair_expanded = cv2.dilate(hair_mask, kernel, iterations=3)
        hair_original = cv2.dilate(hair_mask, kernel, iterations=1)
        
        boundary = cv2.subtract(hair_expanded, hair_original)
        boundary_pixels = image[boundary > 0]
        
        if len(boundary_pixels) > 0:
            # 중간값 사용으로 이상치 제거
            skin_color = np.median(boundary_pixels, axis=0)
        else:
            skin_color = np.array([255, 220, 177])
        
        return skin_color
    
    def _color_correction_v2(self, blended_image, source_image, hair_mask):
        """
        V2 색상 보정
        """
        # 헤어 영역 외부의 색상 분포 분석
        mask_inv = cv2.bitwise_not(hair_mask)
        non_hair_pixels = source_image[mask_inv > 0]
        
        if len(non_hair_pixels) > 0:
            # 원본 이미지의 색상 분포
            source_mean = np.mean(non_hair_pixels, axis=0)
            source_std = np.std(non_hair_pixels, axis=0)
            
            # 블렌딩된 이미지의 색상 분포
            blended_mean = np.mean(blended_image, axis=(0, 1))
            blended_std = np.std(blended_image, axis=(0, 1))
            
            # 색상 보정
            for c in range(3):
                if source_std[c] > 0:
                    blended_image[:, :, c] = (
                        (blended_image[:, :, c] - blended_mean[c]) * (source_std[c] / max(blended_std[c], 1)) +
                        source_mean[c]
                    ).astype(np.uint8)
        
        return blended_image
    
    def _save_result_v2(self, source_path, target_path, result_image):
        """
        V2 결과 저장
        """
        source_name = Path(source_path).stem
        target_name = Path(target_path).stem
        output_name = f"final_v2_{source_name}_to_{target_name}.jpg"
        
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        return str(output_path)
    
    def interactive_change_v2(self):
        """
        V2 대화형 헤어스타일 변경
        """
        print("\n=== V2 헤어스타일 변경 ===")
        
        # 원본 이미지 선택
        original_dir = Path('images/original')
        original_images = list(original_dir.glob('*.jpg')) + list(original_dir.glob('*.jpeg')) + list(original_dir.glob('*.png'))
        
        if not original_images:
            print("❌ 원본 이미지가 없습니다!")
            return
        
        print(f"\n=== 원본 이미지 목록 ({len(original_images)}개) ===")
        for i, img_path in enumerate(original_images, 1):
            print(f"   {i:2d}. {img_path.name}")
        
        # 원본 이미지 선택
        while True:
            try:
                choice = input(f"\n원본 이미지를 선택하세요 (1-{len(original_images)}): ").strip()
                if choice.lower() == 'q':
                    return
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(original_images):
                    original_image = original_images[choice_idx]
                    print(f"✅ 선택된 원본: {original_image.name}")
                    break
                else:
                    print(f"❌ 1-{len(original_images)} 사이의 숫자를 입력하세요.")
            except ValueError:
                print("❌ 올바른 숫자를 입력하세요.")
        
        # 헤어스타일 선택
        hair_dir = Path('images/only_hair')
        hair_images = list(hair_dir.glob('*.jpg'))
        
        if not hair_images:
            print("❌ 헤어 이미지가 없습니다!")
            return
        
        print(f"\n=== 헤어스타일 목록 ({len(hair_images)}개) ===")
        for i, hair_path in enumerate(hair_images, 1):
            print(f"   {i:2d}. {hair_path.stem}")
        
        while True:
            try:
                choice = input(f"\n헤어스타일을 선택하세요 (1-{len(hair_images)}): ").strip()
                if choice.lower() == 'q':
                    return
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(hair_images):
                    target_hair = hair_images[choice_idx]
                    print(f"✅ 선택된 헤어스타일: {target_hair.stem}")
                    break
                else:
                    print(f"❌ 1-{len(hair_images)} 사이의 숫자를 입력하세요.")
            except ValueError:
                print("❌ 올바른 숫자를 입력하세요.")
        
        # V2 헤어스타일 변경 실행
        print(f"\n🎭 V2 헤어스타일 변경 중...")
        print(f"   원본: {original_image.name}")
        print(f"   타겟: {target_hair.name}")
        
        result_path = self.apply_hairstyle_change_v2(
            str(original_image), str(target_hair)
        )
        
        if result_path:
            print(f"\n🎉 V2 헤어스타일 변경 완료!")
            print(f"📁 결과 파일: {Path(result_path).name}")
            print(f"📍 저장 위치: {self.output_dir}")
        else:
            print(f"\n❌ V2 헤어스타일 변경 실패!")

def main():
    """
    메인 함수
    """
    try:
        changer = FinalHairChangerV2()
        
        print("\n=== 최종 헤어스타일 변경 도구 V2 ===")
        print("1. V2 대화형 헤어스타일 변경")
        print("q. 종료")
        
        while True:
            choice = input("\n모드를 선택하세요 (1, q): ").strip().lower()
            
            if choice == 'q':
                print("👋 프로그램을 종료합니다.")
                break
            elif choice == '1':
                changer.interactive_change_v2()
            else:
                print("❌ 1 또는 q를 입력하세요.")
            
            print(f"\n{'='*60}")
    
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 