#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import uuid
import logging
from typing import Optional
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'hair_style_transfer'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'inference', 'face_parsing'))

from improved_hair_transfer import ImprovedHairTransfer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Hair Style Transfer API",
    description="AI 기반 헤어스타일 전송 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
hair_transfer_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# 정적 파일 마운트
app.mount("/images", StaticFiles(directory="images"), name="images")

class TransferRequest(BaseModel):
    source_image_path: str
    reference_image_path: str
    prompt: Optional[str] = "natural hair, seamless blending with face, high quality, detailed hairstyle"
    negative_prompt: Optional[str] = "blurry, low quality, distorted, unnatural, obvious seam, bald, no hair, text, watermark, symbols, characters, artifacts"

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    global hair_transfer_model
    try:
        logger.info(f"GPU 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        logger.info("헤어스타일 전송 모델 로드 중...")
        hair_transfer_model = ImprovedHairTransfer(device=device)
        logger.info("모델 로드 완료!")
        
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        raise Exception("모델 초기화에 실패했습니다.")

@app.get("/", response_class=HTMLResponse)
async def root():
    """웹 인터페이스"""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api")
async def api_root():
    """API 루트 엔드포인트"""
    return {
        "message": "Hair Style Transfer API",
        "version": "1.0.0",
        "status": "running",
        "device": device
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": hair_transfer_model is not None,
        "device": device,
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """이미지 업로드"""
    try:
        # 파일 확장자 검증
        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")
        
        # 고유한 파일명 생성
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = output_dir / unique_filename
        
        # 파일 저장
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"이미지 업로드 완료: {file_path}")
        
        return {
            "filename": unique_filename,
            "file_path": str(file_path),
            "size": len(content)
        }
        
    except Exception as e:
        logger.error(f"이미지 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 업로드 실패: {str(e)}")

@app.post("/transfer")
async def transfer_hair_style(
    source_image: str,
    reference_image: str,
    prompt: Optional[str] = "natural hair, seamless blending with face, high quality, detailed hairstyle",
    negative_prompt: Optional[str] = "blurry, low quality, distorted, unnatural, obvious seam, bald, no hair, text, watermark, symbols, characters, artifacts"
):
    """헤어스타일 전송"""
    try:
        if hair_transfer_model is None:
            raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
        
        # 파일 경로 검증
        source_path = output_dir / source_image
        reference_path = output_dir / reference_image
        
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="소스 이미지를 찾을 수 없습니다.")
        
        if not reference_path.exists():
            raise HTTPException(status_code=404, detail="참조 이미지를 찾을 수 없습니다.")
        
        # 출력 파일명 생성
        output_filename = f"result_{uuid.uuid4()}.png"
        output_path = output_dir / output_filename
        
        logger.info(f"헤어스타일 전송 시작: {source_path} -> {reference_path}")
        
        # 헤어스타일 전송 실행
        success = hair_transfer_model.transfer_hair_style(
            source_image_path=str(source_path),
            reference_image_path=str(reference_path),
            output_path=str(output_path),
            prompt=prompt,
            negative_prompt=negative_prompt
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="헤어스타일 전송에 실패했습니다.")
        
        logger.info(f"헤어스타일 전송 완료: {output_path}")
        
        return {
            "success": True,
            "output_filename": output_filename,
            "output_path": str(output_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"헤어스타일 전송 실패: {e}")
        raise HTTPException(status_code=500, detail=f"헤어스타일 전송 실패: {str(e)}")

@app.get("/download/{filename}")
async def download_result(filename: str):
    """결과 이미지 다운로드"""
    try:
        file_path = output_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="image/png"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 다운로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 다운로드 실패: {str(e)}")

@app.get("/examples")
async def get_example_images():
    """예시 이미지 목록 반환"""
    try:
        examples_dir = Path("images/original")
        if not examples_dir.exists():
            return {"examples": []}
        
        examples = []
        for file_path in examples_dir.glob("*"):
            if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                examples.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size
                })
        
        return {"examples": examples}
        
    except Exception as e:
        logger.error(f"예시 이미지 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"예시 이미지 목록 조회 실패: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
