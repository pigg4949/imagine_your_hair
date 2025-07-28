import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import sys

# inference 폴더를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inference', 'face_parsing'))

from model import BiSeNet

# 입력 및 출력 디렉토리
input_dir = os.path.join('images', 'original')
output_dir = os.path.join('images', 'only_hair_mask')
os.makedirs(output_dir, exist_ok=True)

# BiSeNet 모델 초기화 (19개 클래스)
n_classes = 19
net = BiSeNet(n_classes=n_classes)

# CPU 환경에서 실행
device = torch.device('cpu')
net.to(device)

# 사전학습된 모델 로드 (모델 파일이 있다면)
model_path = os.path.join('inference', 'face_parsing', 'res', 'cp', 'model_final_diss.pth')
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path, map_location=device))
    print(f"모델 로드 완료: {model_path}")
else:
    print(f"모델 파일을 찾을 수 없습니다: {model_path}")
    print("사전학습된 모델을 다운로드해야 합니다:")
    print("1. https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812 에서 모델 파일 다운로드")
    print("2. inference/face_parsing/res/cp/ 폴더를 생성")
    print("3. 다운로드한 파일을 model_final_diss.pth로 이름 변경하여 해당 폴더에 저장")
    print("4. 다시 이 스크립트를 실행하세요.")
    exit(1)

net.eval()

# 이미지 전처리
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 헤어 클래스 (CelebAMask-HQ 데이터셋 기준)
HAIR_CLASS = 17

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, filename)
        
        # 이미지 로드 및 전처리
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        image = img.resize((512, 512), Image.BILINEAR)
        img_tensor = to_tensor(image)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            out = net(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            # 헤어 클래스만 마스크로 추출
            hair_mask = (parsing == HAIR_CLASS).astype(np.uint8) * 255
            
            # 원본 크기로 리사이즈
            hair_mask = cv2.resize(hair_mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            # 마스크 저장
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, hair_mask)
            print(f"헤어 마스크 저장: {out_path}")

print("모든 이미지에 대한 헤어 마스크 생성이 완료되었습니다.")
