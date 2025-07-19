#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import sys
import subprocess
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

class SimpleHairFastGANWrapper:
    def __init__(self):
        """
        간단한 HairFastGAN 래퍼 초기화
        """
        print("🔧 간단한 HairFastGAN 래퍼 초기화 중...")
        
        self.project_dir = Path.cwd()
        self.hairfastgan_dir = self.project_dir / "HairFastGAN"
        self.output_dir = Path('images/hairfastgan_results')
        self.output_dir.mkdir(exist_ok=True)
        
        # HairFastGAN 설치 확인
        if not self.hairfastgan_dir.exists():
            print("   ❌ HairFastGAN이 설치되지 않았습니다.")
            print("   📥 수동 설치가 필요합니다:")
            print("      git clone https://github.com/AIRI-Institute/HairFastGAN")
            print("      cd HairFastGAN")
            print("      pip install -r requirements.txt")
        else:
            print("   ✅ HairFastGAN 설치됨")
        
        print("✅ 간단한 HairFastGAN 래퍼 초기화 완료!")
    
    def check_dependencies(self):
        """
        의존성 확인
        """
        print("🔍 의존성 확인 중...")
        
        required_packages = [
            'torch', 'torchvision', 'opencv-python', 
            'numpy', 'Pillow', 'face_alignment'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} (설치 필요)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n📦 설치 필요한 패키지:")
            for package in missing_packages:
                print(f"   pip install {package}")
        
        return len(missing_packages) == 0
    
    def prepare_images_for_hairfastgan(self, source_image_path, target_hair_path):
        """
        HairFastGAN용 이미지 준비
        """
        print(f"🎯 HairFastGAN용 이미지 준비 중...")
        
        # 이미지 크기 조정 (HairFastGAN은 1024x1024 권장)
        target_size = (1024, 1024)
        
        # 원본 이미지 준비
        source_img = Image.open(source_image_path).convert('RGB')
        source_img_resized = source_img.resize(target_size, Image.LANCZOS)
        
        # 타겟 헤어 이미지 준비
        target_img = Image.open(target_hair_path).convert('RGB')
        target_img_resized = target_img.resize(target_size, Image.LANCZOS)
        
        # 임시 디렉토리 생성
        temp_dir = Path("temp_hairfastgan")
        temp_dir.mkdir(exist_ok=True)
        
        # 이미지 저장
        source_prepared = temp_dir / f"source_{Path(source_image_path).name}"
        target_prepared = temp_dir / f"target_{Path(target_hair_path).name}"
        
        source_img_resized.save(source_prepared)
        target_img_resized.save(target_prepared)
        
        print(f"   ✅ 이미지 준비 완료")
        return str(source_prepared), str(target_prepared)
    
    def run_hairfastgan_simple(self, source_image_path, target_hair_path):
        """
        간단한 HairFastGAN 실행
        """
        print(f"🎭 HairFastGAN 실행 중...")
        print(f"   원본: {Path(source_image_path).name}")
        print(f"   타겟: {Path(target_hair_path).name}")
        
        try:
            # 1. 이미지 준비
            prepared_source, prepared_target = self.prepare_images_for_hairfastgan(
                source_image_path, target_hair_path
            )
            
            # 2. HairFastGAN 실행
            main_script = self.hairfastgan_dir / "main.py"
            
            if not main_script.exists():
                print("   ❌ HairFastGAN main.py를 찾을 수 없습니다.")
                return None
            
            output_name = f"hairfastgan_{Path(source_image_path).stem}_to_{Path(target_hair_path).stem}.png"
            output_path = self.output_dir / output_name
            
            # HairFastGAN 명령어 구성
            cmd = [
                sys.executable, str(main_script),
                "--face_path", prepared_source,
                "--shape_path", prepared_target,
                "--color_path", prepared_target,  # 같은 이미지를 색상으로도 사용
                "--input_dir", str(Path(prepared_source).parent),
                "--result_path", str(output_path)
            ]
            
            print(f"   🔄 HairFastGAN 실행 중...")
            print(f"   명령어: {' '.join(cmd)}")
            
            # 실행
            result = subprocess.run(
                cmd, 
                cwd=self.hairfastgan_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"   ✅ HairFastGAN 실행 완료: {output_path}")
                return str(output_path)
            else:
                print(f"   ❌ HairFastGAN 실행 실패:")
                print(f"   오류: {result.stderr}")
                return None
            
        except Exception as e:
            print(f"   ❌ 오류 발생: {str(e)}")
            return None
    
    def create_demo_script(self):
        """
        HairFastGAN 데모 스크립트 생성
        """
        print("📝 HairFastGAN 데모 스크립트 생성 중...")
        
        demo_script = """
#!/usr/bin/python
# HairFastGAN 데모 스크립트

import sys
import os
from pathlib import Path

# HairFastGAN 경로 추가
hairfastgan_path = Path("HairFastGAN")
if hairfastgan_path.exists():
    sys.path.append(str(hairfastgan_path))

try:
    from hair_swap import HairFast, get_parser
    
    def run_hairfastgan_demo():
        print("🎭 HairFastGAN 데모 실행 중...")
        
        # HairFast 초기화
        args = get_parser().parse_args([])
        hair_fast = HairFast(args)
        
        # 이미지 경로 설정
        face_path = "images/original/gwangju_short.jpg"
        shape_path = "images/only_hair/chawoo_hair.jpg"
        color_path = "images/only_hair/chawoo_hair.jpg"
        
        if not all(Path(p).exists() for p in [face_path, shape_path, color_path]):
            print("❌ 필요한 이미지 파일이 없습니다.")
            return
        
        # 이미지 로드
        from PIL import Image
        face_img = Image.open(face_path).convert('RGB')
        shape_img = Image.open(shape_path).convert('RGB')
        color_img = Image.open(color_path).convert('RGB')
        
        # HairFast 실행
        result = hair_fast(face_img, shape_img, color_img)
        
        # 결과 저장
        output_path = "images/hairfastgan_results/demo_result.png"
        Path(output_path).parent.mkdir(exist_ok=True)
        result.save(output_path)
        
        print(f"✅ 데모 완료: {output_path}")
    
    if __name__ == "__main__":
        run_hairfastgan_demo()
        
except ImportError as e:
    print(f"❌ HairFastGAN 모듈을 불러올 수 없습니다: {e}")
    print("📦 의존성 설치가 필요합니다:")
    print("   cd HairFastGAN")
    print("   pip install -r requirements.txt")
except Exception as e:
    print(f"❌ 오류 발생: {e}")
"""
        
        with open("hairfastgan_demo.py", "w", encoding="utf-8") as f:
            f.write(demo_script)
        
        print("   ✅ 데모 스크립트 생성 완료: hairfastgan_demo.py")
    
    def interactive_change(self):
        """
        대화형 HairFastGAN 헤어스타일 변경
        """
        print("\n=== HairFastGAN 헤어스타일 변경 ===")
        
        # 의존성 확인
        if not self.check_dependencies():
            print("\n⚠️ 일부 의존성이 누락되었습니다.")
            print("HairFastGAN을 사용하기 전에 필요한 패키지를 설치하세요.")
            return
        
        # HairFastGAN 설치 확인
        if not self.hairfastgan_dir.exists():
            print("\n❌ HairFastGAN이 설치되지 않았습니다.")
            print("먼저 HairFastGAN을 설치하세요:")
            print("   git clone https://github.com/AIRI-Institute/HairFastGAN")
            return
        
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
        
        # HairFastGAN 실행
        print(f"\n🎭 HairFastGAN 헤어스타일 변경 중...")
        print(f"   원본: {original_image.name}")
        print(f"   타겟: {target_hair.name}")
        
        result_path = self.run_hairfastgan_simple(
            str(original_image), str(target_hair)
        )
        
        if result_path:
            print(f"\n🎉 HairFastGAN 헤어스타일 변경 완료!")
            print(f"📁 결과 파일: {Path(result_path).name}")
            print(f"📍 저장 위치: {self.output_dir}")
        else:
            print(f"\n❌ 헤어스타일 변경 실패!")
            print("💡 데모 스크립트를 생성하여 수동으로 실행해보세요.")

def main():
    """
    메인 함수
    """
    try:
        wrapper = SimpleHairFastGANWrapper()
        
        print("\n=== HairFastGAN 헤어스타일 변경 도구 ===")
        print("1. 대화형 HairFastGAN 헤어스타일 변경")
        print("2. 의존성 확인")
        print("3. 데모 스크립트 생성")
        print("4. HairFastGAN 정보")
        print("q. 종료")
        
        while True:
            choice = input("\n모드를 선택하세요 (1-4, q): ").strip().lower()
            
            if choice == 'q':
                print("👋 프로그램을 종료합니다.")
                break
            elif choice == '1':
                wrapper.interactive_change()
            elif choice == '2':
                wrapper.check_dependencies()
            elif choice == '3':
                wrapper.create_demo_script()
            elif choice == '4':
                print("\n=== HairFastGAN 정보 ===")
                print("📄 논문: NeurIPS 2024")
                print("🔗 GitHub: https://github.com/AIRI-Institute/HairFastGAN")
                print("🎯 특징:")
                print("   - 실시간 처리 (1초 미만)")
                print("   - 고해상도 생성")
                print("   - 포즈 차이 적응")
                print("   - 정확한 헤어스타일 전송")
                print("   - 색상 분리 전송")
                print("\n📦 설치 방법:")
                print("   git clone https://github.com/AIRI-Institute/HairFastGAN")
                print("   cd HairFastGAN")
                print("   pip install -r requirements.txt")
            else:
                print("❌ 1-4 또는 q를 입력하세요.")
            
            print(f"\n{'='*60}")
    
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 