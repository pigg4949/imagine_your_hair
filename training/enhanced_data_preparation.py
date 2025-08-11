#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import argparse
from pathlib import Path
import shutil
from typing import List, Tuple, Dict
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataPreparation:
    def __init__(self, input_dir: str, output_dir: str, target_size: Tuple[int, int] = (512, 512)):
        """
        향상된 데이터 준비 클래스 초기화
        
        Args:
            input_dir: 입력 이미지 디렉토리
            output_dir: 출력 디렉토리
            target_size: 목표 이미지 크기
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 통계 정보 저장
        self.stats = {
            "total_images": 0,
            "processed_images": 0,
            "rejected_images": 0,
            "quality_scores": [],
            "face_sizes": [],
            "processing_time": 0
        }
        
    def prepare_training_data(
        self, 
        min_face_size: int = 100, 
        quality_threshold: float = 0.7,
        enhance_quality: bool = True,
        create_preview: bool = True
    ):
        """
        향상된 학습 데이터 준비
        
        Args:
            min_face_size: 최소 얼굴 크기
            quality_threshold: 이미지 품질 임계값
            enhance_quality: 이미지 품질 향상 여부
            create_preview: 미리보기 생성 여부
        """
        import time
        start_time = time.time()
        
        logger.info("향상된 학습 데이터 준비 시작...")
        
        # 지원하는 이미지 확장자
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # 입력 이미지 파일들 찾기
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.error(f"입력 디렉토리에서 이미지를 찾을 수 없습니다: {self.input_dir}")
            return
        
        self.stats["total_images"] = len(image_files)
        logger.info(f"총 {len(image_files)}개의 이미지 파일 발견")
        
        # 얼굴 검출기 초기화
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        processed_count = 0
        rejected_count = 0
        
        for image_file in image_files:
            try:
                logger.info(f"처리 중: {image_file.name}")
                
                # 이미지 로드
                image = cv2.imread(str(image_file))
                if image is None:
                    logger.warning(f"이미지를 로드할 수 없습니다: {image_file}")
                    rejected_count += 1
                    continue
                
                # BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 얼굴 검출
                faces = face_cascade.detectMultiScale(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(min_face_size, min_face_size)
                )
                
                if len(faces) == 0:
                    logger.warning(f"얼굴이 검출되지 않았습니다: {image_file}")
                    rejected_count += 1
                    continue
                
                # 가장 큰 얼굴 선택
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # 얼굴 크기 통계 저장
                self.stats["face_sizes"].append(w * h)
                
                # 얼굴 영역 확장 (헤어 포함)
                margin = int(max(w, h) * 0.4)  # 헤어를 더 많이 포함
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image.shape[1], x + w + margin)
                y2 = min(image.shape[0], y + h + margin)
                
                # 얼굴 영역 크롭
                face_crop = image_rgb[y1:y2, x1:x2]
                
                # 이미지 품질 검사
                quality_score = self._calculate_quality_score(face_crop)
                self.stats["quality_scores"].append(quality_score)
                
                if quality_score < quality_threshold:
                    logger.warning(f"이미지 품질이 낮습니다: {image_file} (점수: {quality_score:.3f})")
                    rejected_count += 1
                    continue
                
                # 이미지 품질 향상 (선택사항)
                if enhance_quality:
                    face_crop = self._enhance_image_quality(face_crop)
                
                # 리사이즈
                face_crop_resized = cv2.resize(face_crop, self.target_size)
                
                # 출력 파일명 생성
                output_filename = f"training_{processed_count:04d}.jpg"
                output_path = self.output_dir / output_filename
                
                # 이미지 저장
                cv2.imwrite(str(output_path), cv2.cvtColor(face_crop_resized, cv2.COLOR_RGB2BGR))
                
                processed_count += 1
                logger.info(f"처리 완료: {image_file.name} -> {output_filename} (품질: {quality_score:.3f})")
                
            except Exception as e:
                logger.error(f"이미지 처리 중 오류 발생: {image_file} - {e}")
                rejected_count += 1
                continue
        
        # 통계 업데이트
        self.stats["processed_images"] = processed_count
        self.stats["rejected_images"] = rejected_count
        self.stats["processing_time"] = time.time() - start_time
        
        # 결과 출력
        self._print_statistics()
        
        # 미리보기 생성
        if create_preview:
            self._create_preview()
        
        # 통계 저장
        self._save_statistics()
        
        logger.info(f"데이터 준비 완료! 총 {processed_count}개의 이미지가 처리되었습니다.")
        logger.info(f"출력 디렉토리: {self.output_dir}")
    
    def _calculate_quality_score(self, image: np.ndarray) -> float:
        """
        이미지 품질 점수를 계산합니다.
        
        Args:
            image: 검사할 이미지
            
        Returns:
            품질 점수 (0-1)
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. 선명도 (라플라시안 분산)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)
        
        # 2. 대비
        contrast = gray.std()
        contrast_score = min(contrast / 50.0, 1.0)
        
        # 3. 밝기 (너무 어둡거나 밝지 않게)
        brightness = gray.mean()
        brightness_score = 1.0 - abs(brightness - 128) / 128.0
        brightness_score = max(0.0, brightness_score)
        
        # 4. 노이즈 (가우시안 블러로 노이즈 측정)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blurred)
        noise_score = 1.0 - min(noise.mean() / 10.0, 1.0)
        
        # 종합 품질 점수
        quality_score = (sharpness_score * 0.4 + 
                        contrast_score * 0.3 + 
                        brightness_score * 0.2 + 
                        noise_score * 0.1)
        
        return quality_score
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 품질을 향상시킵니다.
        
        Args:
            image: 향상할 이미지
            
        Returns:
            향상된 이미지
        """
        # PIL Image로 변환
        pil_image = Image.fromarray(image)
        
        # 1. 대비 향상
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # 2. 선명도 향상
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # 3. 밝기 조정
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.05)
        
        # numpy 배열로 변환
        enhanced_image = np.array(pil_image)
        
        return enhanced_image
    
    def _print_statistics(self):
        """통계 정보를 출력합니다."""
        print("\n" + "="*50)
        print("📊 데이터 전처리 통계")
        print("="*50)
        print(f"총 이미지 수: {self.stats['total_images']}")
        print(f"처리된 이미지: {self.stats['processed_images']}")
        print(f"거부된 이미지: {self.stats['rejected_images']}")
        print(f"처리 시간: {self.stats['processing_time']:.2f}초")
        
        if self.stats['quality_scores']:
            print(f"평균 품질 점수: {np.mean(self.stats['quality_scores']):.3f}")
            print(f"최고 품질 점수: {np.max(self.stats['quality_scores']):.3f}")
            print(f"최저 품질 점수: {np.min(self.stats['quality_scores']):.3f}")
        
        if self.stats['face_sizes']:
            print(f"평균 얼굴 크기: {np.mean(self.stats['face_sizes']):.0f} 픽셀")
        
        print("="*50)
    
    def _create_preview(self):
        """처리된 이미지들의 미리보기를 생성합니다."""
        try:
            # 처리된 이미지들 찾기
            processed_images = list(self.output_dir.glob("*.jpg"))
            
            if len(processed_images) == 0:
                logger.warning("미리보기를 생성할 이미지가 없습니다.")
                return
            
            # 최대 9개 이미지로 미리보기 생성
            preview_images = processed_images[:9]
            
            # 미리보기 그리드 생성
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            fig.suptitle('처리된 이미지 미리보기', fontsize=16)
            
            for i, image_path in enumerate(preview_images):
                row = i // 3
                col = i % 3
                
                # 이미지 로드
                image = cv2.imread(str(image_path))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                axes[row, col].imshow(image_rgb)
                axes[row, col].set_title(f"{image_path.name}")
                axes[row, col].axis('off')
            
            # 빈 서브플롯 숨기기
            for i in range(len(preview_images), 9):
                row = i // 3
                col = i % 3
                axes[row, col].axis('off')
            
            # 미리보기 저장
            preview_path = self.output_dir / "preview.png"
            plt.tight_layout()
            plt.savefig(preview_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"미리보기 생성 완료: {preview_path}")
            
        except Exception as e:
            logger.error(f"미리보기 생성 실패: {e}")
    
    def _save_statistics(self):
        """통계 정보를 JSON 파일로 저장합니다."""
        stats_file = self.output_dir / "processing_stats.json"
        
        # numpy 배열을 리스트로 변환
        stats_to_save = {
            "total_images": self.stats["total_images"],
            "processed_images": self.stats["processed_images"],
            "rejected_images": self.stats["rejected_images"],
            "processing_time": self.stats["processing_time"],
            "quality_scores": [float(score) for score in self.stats["quality_scores"]],
            "face_sizes": [int(size) for size in self.stats["face_sizes"]],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"통계 정보 저장 완료: {stats_file}")
    
    def create_validation_split(self, validation_ratio: float = 0.2):
        """
        검증 데이터셋을 분리합니다.
        
        Args:
            validation_ratio: 검증 데이터 비율
        """
        logger.info("검증 데이터셋 분리 중...")
        
        # 학습 데이터 파일들
        training_files = list(self.output_dir.glob("*.jpg"))
        
        if not training_files:
            logger.warning("학습 데이터가 없습니다.")
            return
        
        # 검증 데이터 개수 계산
        num_validation = int(len(training_files) * validation_ratio)
        
        # 랜덤하게 검증 데이터 선택
        import random
        random.shuffle(training_files)
        validation_files = training_files[:num_validation]
        
        # 검증 디렉토리 생성
        validation_dir = self.output_dir / "validation"
        validation_dir.mkdir(exist_ok=True)
        
        # 검증 파일들 이동
        for file in validation_files:
            shutil.move(str(file), str(validation_dir / file.name))
        
        logger.info(f"검증 데이터 분리 완료: {len(validation_files)}개 파일")
        logger.info(f"검증 디렉토리: {validation_dir}")

def main():
    parser = argparse.ArgumentParser(description="향상된 학습 데이터 준비")
    parser.add_argument("--input_dir", type=str, required=True, help="입력 이미지 디렉토리")
    parser.add_argument("--output_dir", type=str, default="training_data", help="출력 디렉토리")
    parser.add_argument("--target_size", type=int, nargs=2, default=[512, 512], help="목표 이미지 크기")
    parser.add_argument("--min_face_size", type=int, default=100, help="최소 얼굴 크기")
    parser.add_argument("--quality_threshold", type=float, default=0.7, help="이미지 품질 임계값")
    parser.add_argument("--enhance_quality", action="store_true", help="이미지 품질 향상")
    parser.add_argument("--create_preview", action="store_true", help="미리보기 생성")
    parser.add_argument("--validation_ratio", type=float, default=0.2, help="검증 데이터 비율")
    
    args = parser.parse_args()
    
    # 데이터 준비
    data_prep = EnhancedDataPreparation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size)
    )
    
    # 학습 데이터 준비
    data_prep.prepare_training_data(
        min_face_size=args.min_face_size,
        quality_threshold=args.quality_threshold,
        enhance_quality=args.enhance_quality,
        create_preview=args.create_preview
    )
    
    # 검증 데이터 분리
    data_prep.create_validation_split(validation_ratio=args.validation_ratio)

if __name__ == "__main__":
    main()
