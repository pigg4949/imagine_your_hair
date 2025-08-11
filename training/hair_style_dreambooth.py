#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import Dataset
import numpy as np
from PIL import Image
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = get_logger(__name__)

class HairStyleDreamBooth:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        training_data_dir: str = "training_data",
        output_dir: str = "trained_hair_models",
        hair_style_token: str = "hst",  # hair style token
        class_prompt: str = "a photo of a person with hair",
        num_class_images: int = 50,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        max_train_steps: int = 1000,
        learning_rate: float = 1e-6,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 27,
        lora_dropout: float = 0.0,
        mixed_precision: str = "fp16",
        seed: int = 42,
        device: str = "cuda"
    ):
        """
        헤어스타일 특화 DreamBooth 학습기
        
        Args:
            pretrained_model_name_or_path: 사전학습된 모델 경로
            training_data_dir: 학습 데이터 디렉토리
            output_dir: 출력 디렉토리
            hair_style_token: 헤어스타일 토큰 (예: "hst")
            class_prompt: 클래스 프롬프트
            num_class_images: 클래스 이미지 수
            train_batch_size: 학습 배치 크기
            gradient_accumulation_steps: 그래디언트 누적 스텝
            max_train_steps: 최대 학습 스텝
            learning_rate: 학습률
            lr_scheduler: 학습률 스케줄러
            lr_warmup_steps: 워밍업 스텝
            use_lora: LoRA 사용 여부
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            mixed_precision: 혼합 정밀도
            seed: 랜덤 시드
            device: 사용할 디바이스
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.training_data_dir = training_data_dir
        self.output_dir = output_dir
        self.hair_style_token = hair_style_token
        self.class_prompt = class_prompt
        self.num_class_images = num_class_images
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_train_steps = max_train_steps
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.mixed_precision = mixed_precision
        self.seed = seed
        self.device = device
        
        # 디렉토리 생성
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Accelerator 초기화
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        
        # 랜덤 시드 설정
        set_seed(self.seed)
        
        # 모델 로드
        self._load_models()
        
    def _load_models(self):
        """모델들을 로드합니다."""
        logger.info("헤어스타일 DreamBooth 모델 로딩 중...")
        
        # 토크나이저와 텍스트 인코더
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        
        # 새로운 토큰 추가 (헤어스타일 토큰)
        num_added_tokens = self.tokenizer.add_tokens(self.hair_style_token)
        logger.info(f"새로운 토큰 추가: {self.hair_style_token}")
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
        )
        
        # 텍스트 인코더의 임베딩 레이어 크기 조정
        if num_added_tokens > 0:
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        
        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
        )
        
        # UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="unet",
        )
        
        # LoRA 설정
        if self.use_lora:
            self._setup_lora()
        
        # 스케줄러
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        
        logger.info("모델 로딩 완료!")
    
    def _setup_lora(self):
        """LoRA 설정"""
        logger.info("LoRA 설정 중...")
        
        # LoRA 어텐션 프로세서 설정
        lora_attn_procs = {}
        for name, module in self.unet.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                name = name.replace(".", "_")
                if name.startswith("transformer_blocks"):
                    name = name.replace("transformer_blocks_", "")
                    name = name.replace("_attn1", "")
                    name = name.replace("_attn2", "")
                    name = name.replace("_", ".")
                    lora_attn_procs[f"{name}.to_q"] = LoRAAttnProcessor(
                        hidden_size=module.hidden_size,
                        cross_attention_dim=None,
                        rank=self.lora_r,
                        dropout=self.lora_dropout,
                    )
                    lora_attn_procs[f"{name}.to_k"] = LoRAAttnProcessor(
                        hidden_size=module.hidden_size,
                        cross_attention_dim=None,
                        rank=self.lora_r,
                        dropout=self.lora_dropout,
                    )
                    lora_attn_procs[f"{name}.to_v"] = LoRAAttnProcessor(
                        hidden_size=module.hidden_size,
                        cross_attention_dim=None,
                        rank=self.lora_r,
                        dropout=self.lora_dropout,
                    )
                    lora_attn_procs[f"{name}.to_out.0"] = LoRAAttnProcessor(
                        hidden_size=module.hidden_size,
                        cross_attention_dim=None,
                        rank=self.lora_r,
                        dropout=self.lora_dropout,
                    )
        
        self.unet.set_attn_processor(lora_attn_procs)
        logger.info("LoRA 설정 완료!")
    
    def prepare_dataset(self):
        """데이터셋을 준비합니다."""
        logger.info("헤어스타일 데이터셋 준비 중...")
        
        # 학습 이미지 로드
        training_images = []
        training_image_paths = []
        
        training_dir = Path(self.training_data_dir)
        
        # 학습 이미지 파일들 찾기
        for image_path in training_dir.glob("*.jpg"):
            training_image_paths.append(str(image_path))
        for image_path in training_dir.glob("*.png"):
            training_image_paths.append(str(image_path))
        
        if len(training_image_paths) == 0:
            raise ValueError(f"학습 이미지를 찾을 수 없습니다: {self.training_data_dir}")
        
        for image_path in training_image_paths:
            image = Image.open(image_path).convert("RGB")
            training_images.append(image)
        
        logger.info(f"학습 이미지 {len(training_images)}개 로드됨")
        
        # 인스턴스 프롬프트 생성 (헤어스타일 토큰 사용)
        instance_prompt = f"a photo of {self.hair_style_token} person"
        
        # 데이터셋 생성
        self.dataset = Dataset.from_dict({
            "image": training_images,
            "prompt": [instance_prompt] * len(training_images)
        })
        
        # 전처리 함수
        def transform_images(examples):
            images = [image.convert("RGB") for image in examples["image"]]
            examples["pixel_values"] = [self._preprocess_image(image) for image in images]
            return examples
        
        self.dataset.set_transform(transform_images)
        logger.info("데이터셋 준비 완료!")
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """이미지를 전처리합니다."""
        # 리사이즈
        image = image.resize((512, 512))
        
        # 텐서 변환 및 정규화
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image = (image - 0.5) * 2.0
        
        return image
    
    def train(self):
        """모델을 학습합니다."""
        logger.info("헤어스타일 DreamBooth 학습 시작...")
        
        # 데이터셋 준비
        self.prepare_dataset()
        
        # 옵티마이저
        optimizer_cls = torch.optim.AdamW
        
        # 학습 가능한 파라미터
        if self.use_lora:
            params_to_optimize = []
            for name, param in self.unet.named_parameters():
                if "lora" in name:
                    params_to_optimize.append(param)
        else:
            params_to_optimize = list(self.unet.parameters())
        
        optimizer = optimizer_cls(
            params_to_optimize,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
        
        # 학습률 스케줄러
        if self.lr_scheduler == "constant":
            lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        elif self.lr_scheduler == "linear":
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.max_train_steps
            )
        else:
            raise ValueError(f"지원하지 않는 스케줄러: {self.lr_scheduler}")
        
        # Accelerator로 모델들 준비
        self.unet, self.text_encoder, self.vae, optimizer, lr_scheduler = self.accelerator.prepare(
            self.unet, self.text_encoder, self.vae, optimizer, lr_scheduler
        )
        
        # 학습 루프
        progress_bar = self.accelerator.init_progress_bar()
        global_step = 0
        
        for epoch in range(self.max_train_steps):
            self.unet.train()
            
            for batch in self.dataset:
                with self.accelerator.accumulate(self.unet):
                    # 배치 데이터 준비
                    pixel_values = torch.stack(batch["pixel_values"]).to(self.device)
                    prompts = batch["prompt"]
                    
                    # 텍스트 인코딩
                    input_ids = self.tokenizer(
                        prompts,
                        padding="max_length",
                        truncation=True,
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(self.device)
                    
                    # 텍스트 임베딩
                    with torch.no_grad():
                        encoder_hidden_states = self.text_encoder(input_ids)[0]
                    
                    # 노이즈 추가
                    noise = torch.randn_like(pixel_values)
                    bsz = pixel_values.shape[0]
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=pixel_values.device)
                    timesteps = timesteps.long()
                    
                    noisy_pixel_values = self.noise_scheduler.add_noise(pixel_values, noise, timesteps)
                    
                    # 예측
                    noise_pred = self.unet(noisy_pixel_values, timesteps, encoder_hidden_states).sample
                    
                    # 손실 계산
                    loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                    
                    # 역전파
                    self.accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                progress_bar.update(1)
                global_step += 1
                
                if global_step >= self.max_train_steps:
                    break
            
            if global_step >= self.max_train_steps:
                break
        
        # 모델 저장
        self.save_model()
        logger.info("헤어스타일 DreamBooth 학습 완료!")
    
    def save_model(self):
        """학습된 모델을 저장합니다."""
        logger.info("헤어스타일 모델 저장 중...")
        
        # LoRA 가중치 저장
        if self.use_lora:
            self.unet.save_attn_procs(self.output_dir)
        
        # 토크나이저 저장
        self.tokenizer.save_pretrained(self.output_dir)
        
        # 설정 저장
        config = {
            "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
            "hair_style_token": self.hair_style_token,
            "class_prompt": self.class_prompt,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "training_steps": self.max_train_steps,
            "learning_rate": self.learning_rate,
        }
        
        with open(os.path.join(self.output_dir, "hair_style_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"헤어스타일 모델이 {self.output_dir}에 저장되었습니다.")
        logger.info(f"사용법: '{self.hair_style_token} person with [헤어스타일]' 프롬프트로 생성")

def main():
    parser = argparse.ArgumentParser(description="헤어스타일 DreamBooth 학습")
    parser.add_argument("--training_data_dir", type=str, required=True, help="학습 데이터 디렉토리")
    parser.add_argument("--output_dir", type=str, default="trained_hair_models", help="출력 디렉토리")
    parser.add_argument("--hair_style_token", type=str, default="hst", help="헤어스타일 토큰")
    parser.add_argument("--class_prompt", type=str, default="a photo of a person with hair", help="클래스 프롬프트")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="최대 학습 스텝")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="학습률")
    parser.add_argument("--use_lora", action="store_true", help="LoRA 사용")
    parser.add_argument("--device", type=str, default="cuda", help="사용할 디바이스")
    
    args = parser.parse_args()
    
    trainer = HairStyleDreamBooth(
        training_data_dir=args.training_data_dir,
        output_dir=args.output_dir,
        hair_style_token=args.hair_style_token,
        class_prompt=args.class_prompt,
        max_train_steps=args.max_train_steps,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        device=args.device
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
