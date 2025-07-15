# 간단한 얼굴 감지 테스트 스크립트
import cv2
import mediapipe as mp
import os

def test_face_detection():
    print("=== 얼굴 감지 테스트 시작 ===")
    
    # 미디어파이프 설정
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # 현재 디렉토리 확인
    current_dir = os.getcwd()
    print(f"현재 디렉토리: {current_dir}")
    
    # 이미지 파일 찾기
    image_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print("❌ 이미지 파일을 찾을 수 없습니다!")
        print("images 폴더에 .jpg, .jpeg, .png 파일을 넣어주세요.")
        return
    
    print(f"발견된 이미지 파일들: {image_files}")
    
    # 첫 번째 이미지로 테스트
    image_path = image_files[0]
    print(f"테스트할 이미지: {image_path}")
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 얼굴 감지
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        results = face_detection.process(image_rgb)
        
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
            print(f"✅ 얼굴 {len(results.detections)}개가 감지되었습니다!")
        else:
            print("❌ 얼굴이 감지되지 않았습니다.")
    
    # 결과 저장
    output_path = 'face_detection_result.jpg'
    cv2.imwrite(output_path, image)
    print(f"결과 이미지 저장: {output_path}")
    
    # 결과 표시 (3초)
    cv2.imshow('Face Detection Result', image)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    print("=== 테스트 완료 ===")

if __name__ == "__main__":
    test_face_detection() 