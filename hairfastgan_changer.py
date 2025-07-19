#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import sys
import subprocess
from pathlib import Path
from PIL import Image

class HairFastGANChanger:
    def __init__(self):
        """
        HairFastGAN 헤어스타일 변경기 초기화
        """
        print("🔧 HairFastGAN 헤어스타일 변경기 초기화 중...")
        
        self.project_dir = Path.cwd()
        self.hairfastgan_dir = self.project_dir / "HairFastGAN"
        self.output_dir = Path('images/hairfastgan_results')
        self.output_dir.mkdir(exist_ok=True)
        
        # HairFastGAN 설치 확인
        if not self.hairfastgan_dir.exists():
            print("   ❌ HairFastGAN이 설치되지 않았습니다.")
            print("   📥 설치 명령어:")
            print("      git clone https://github.com/AIRI-Institute/HairFastGAN")
            return
        
        print("   ✅ HairFastGAN 설치됨")
        
        # main.py 확인
        main_script = self.hairfastgan_dir / "main.py"
        if not main_script.exists():
            print("   ❌ HairFastGAN main.py를 찾을 수 없습니다.")
            return
        
        print("   ✅ HairFastGAN main.py 확인됨")
        print("✅ HairFastGAN 헤어스타일 변경기 초기화 완료!")
    
    def prepare_image_for_hairfastgan(self, image_path, output_name):
        """
        HairFastGAN용 이미지 준비 (1024x1024로 리사이즈)
        """
        try:
            # 이미지 로드
            img = Image.open(image_path).convert('RGB')
            
            # 1024x1024로 리사이즈
            img_resized = img.resize((1024, 1024), Image.LANCZOS)
            
            # 임시 디렉토리 생성
            temp_dir = Path("temp_hairfastgan")
            temp_dir.mkdir(exist_ok=True)
            
            # 저장
            output_path = temp_dir / output_name
            img_resized.save(output_path)
            
            return str(output_path)
            
        except Exception as e:
            print(f"   ❌ 이미지 준비 실패: {e}")
            return None
    
    def apply_hairstyle_change(self, source_image_path, target_hair_path):
        """
        HairFastGAN을 사용한 헤어스타일 변경
        """
        print(f"🎭 HairFastGAN 헤어스타일 변경 시작...")
        print(f"   원본: {Path(source_image_path).name}")
        print(f"   타겟: {Path(target_hair_path).name}")
        
        try:
            # 1. 이미지 준비
            print("   🔄 이미지 준비 중...")
            prepared_source = self.prepare_image_for_hairfastgan(
                source_image_path, f"source_{Path(source_image_path).name}"
            )
            prepared_target = self.prepare_image_for_hairfastgan(
                target_hair_path, f"target_{Path(target_hair_path).name}"
            )
            
            if not prepared_source or not prepared_target:
                print("   ❌ 이미지 준비 실패")
                return None
            
            # 2. HairFastGAN 실행
            print("   🔄 HairFastGAN 실행 중...")
            main_script = self.hairfastgan_dir / "main.py"
            
            output_name = f"hairfastgan_{Path(source_image_path).stem}_to_{Path(target_hair_path).stem}.png"
            output_path = self.output_dir / output_name
            
            # HairFastGAN 명령어
            cmd = [
                sys.executable, str(main_script),
                "--face_path", prepared_source,
                "--shape_path", prepared_target,
                "--color_path", prepared_target,
                "--input_dir", str(Path(prepared_source).parent),
                "--result_path", str(output_path)
            ]
            
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
    
    def interactive_change(self):
        """
        대화형 헤어스타일 변경
        """
        print("\n=== HairFastGAN 헤어스타일 변경 ===")
        
        # HairFastGAN 설치 확인
        if not self.hairfastgan_dir.exists():
            print("❌ HairFastGAN이 설치되지 않았습니다.")
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
        
        result_path = self.apply_hairstyle_change(
            str(original_image), str(target_hair)
        )
        
        if result_path:
            print(f"\n🎉 HairFastGAN 헤어스타일 변경 완료!")
            print(f"📁 결과 파일: {Path(result_path).name}")
            print(f"📍 저장 위치: {self.output_dir}")
        else:
            print(f"\n❌ 헤어스타일 변경 실패!")

def main():
    """
    메인 함수
    """
    try:
        changer = HairFastGANChanger()
        
        print("\n=== HairFastGAN 헤어스타일 변경 도구 ===")
        print("1. 대화형 헤어스타일 변경")
        print("q. 종료")
        
        while True:
            choice = input("\n모드를 선택하세요 (1, q): ").strip().lower()
            
            if choice == 'q':
                print("👋 프로그램을 종료합니다.")
                break
            elif choice == '1':
                changer.interactive_change()
            else:
                print("❌ 1 또는 q를 입력하세요.")
            
            print(f"\n{'='*60}")
    
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 