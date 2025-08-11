#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import shutil
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(self, input_dir: str, output_dir: str, target_size: Tuple[int, int] = (512, 512)):
        """
        데이터 준비 클래스 초기화
        
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
        
    def prepare_training_data(self, min_face_size: int = 100, quality_threshold: float = 0.7):
        """
        학습 데이터를 준비합니다.
        
        Args:
            min_face_size: 최소 얼굴 크기
            quality_threshold: 이미지 품질 임계값
        """
        logger.info("학습 데이터 준비 시작...")
        
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
        
        logger.info(f"총 {len(image_files)}개의 이미지 파일 발견")
        
        # 얼굴 검출기 초기화
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        processed_count = 0
        for image_file in image_files:
            try:
                # 이미지 로드
                image = cv2.imread(str(image_file))
                if image is None:
                    logger.warning(f"이미지를 로드할 수 없습니다: {image_file}")
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
                    continue
                
                # 가장 큰 얼굴 선택
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # 얼굴 영역 확장 (헤어 포함)
                margin = int(max(w, h) * 0.3)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image.shape[1], x + w + margin)
                y2 = min(image.shape[0], y + h + margin)
                
                # 얼굴 영역 크롭
                face_crop = image_rgb[y1:y2, x1:x2]
                
                # 이미지 품질 검사
                if not self._check_image_quality(face_crop, quality_threshold):
                    logger.warning(f"이미지 품질이 낮습니다: {image_file}")
                    continue
                
                # 리사이즈
                face_crop_resized = cv2.resize(face_crop, self.target_size)
                
                # 출력 파일명 생성
                output_filename = f"training_{processed_count:04d}.jpg"
                output_path = self.output_dir / output_filename
                
                # 이미지 저장
                cv2.imwrite(str(output_path), cv2.cvtColor(face_crop_resized, cv2.COLOR_RGB2BGR))
                
                processed_count += 1
                logger.info(f"처리 완료: {image_file.name} -> {output_filename}")
                
            except Exception as e:
                logger.error(f"이미지 처리 중 오류 발생: {image_file} - {e}")
                continue
        
        logger.info(f"데이터 준비 완료! 총 {processed_count}개의 이미지가 처리되었습니다.")
        logger.info(f"출력 디렉토리: {self.output_dir}")
    
    def _check_image_quality(self, image: np.ndarray, threshold: float) -> bool:
        """
        이미지 품질을 검사합니다.
        
        Args:
            image: 검사할 이미지
            threshold: 품질 임계값
            
        Returns:
            품질이 기준을 만족하면 True
        """
        # 라플라시안 분산으로 선명도 측정
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 대비 측정
        contrast = gray.std()
        
        # 밝기 측정
        brightness = gray.mean()
        
        # 품질 점수 계산
        quality_score = (laplacian_var / 1000.0) * (contrast / 50.0) * (brightness / 128.0)
        
        return quality_score > threshold
    
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
    parser = argparse.ArgumentParser(description="학습 데이터 준비")
    parser.add_argument("--input_dir", type=str, required=True, help="입력 이미지 디렉토리")
    parser.add_argument("--output_dir", type=str, default="training_data", help="출력 디렉토리")
    parser.add_argument("--target_size", type=int, nargs=2, default=[512, 512], help="목표 이미지 크기")
    parser.add_argument("--min_face_size", type=int, default=100, help="최소 얼굴 크기")
    parser.add_argument("--quality_threshold", type=float, default=0.7, help="이미지 품질 임계값")
    parser.add_argument("--validation_ratio", type=float, default=0.2, help="검증 데이터 비율")
    
    args = parser.parse_args()
    
    # 데이터 준비
    data_prep = DataPreparation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size)
    )
    
    # 학습 데이터 준비
    data_prep.prepare_training_data(
        min_face_size=args.min_face_size,
        quality_threshold=args.quality_threshold
    )
    
    # 검증 데이터 분리
    data_prep.create_validation_split(validation_ratio=args.validation_ratio)

if __name__ == "__main__":
    main()
