import cv2
import os
import numpy as np
from PIL import Image

input_dir = os.path.join('images', 'original')
mask_dir = os.path.join('images', 'only_hair_mask')
output_dir = os.path.join('images', 'only_hair')
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        
        # 출력 파일명을 PNG로 변경
        name_without_ext = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, f"{name_without_ext}.png")
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print(f"이미지 또는 마스크 로드 실패: {filename}")
            continue
        
        # BGR을 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 마스크를 0-1 범위로 정규화
        mask_normalized = mask.astype(np.float32) / 255.0
        
        # RGBA 이미지 생성 (RGB + 알파 채널)
        rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = image_rgb  # RGB 채널
        rgba_image[:, :, 3] = (mask_normalized * 255).astype(np.uint8)  # 알파 채널
        
        # PIL Image로 변환하여 PNG로 저장
        pil_image = Image.fromarray(rgba_image, 'RGBA')
        pil_image.save(out_path, 'PNG')
        
        print(f"투명 배경 헤어 추출 이미지 저장: {out_path}")
