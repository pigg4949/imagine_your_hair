# 얼굴 탐지 서비스 만들기
import cv2
import mediapipe as mp

# 미디어파이프 설정하기
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 이미지 불러오기
import os

# 현재 스크립트의 디렉토리를 기준으로 상대 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
image_path = os.path.join(project_root, 'images', 'gwangju_long.jpg')
image = cv2.imread(image_path)
if image is None:
    print(f"경고: {image_path} 이미지를 찾을 수 없습니다.")
    print("대체 이미지를 사용합니다...")
    
    # 대체 이미지 경로들 시도
    alternative_paths = [
        os.path.join(project_root, 'images', 'gwangju_short.jpg'),
        'gwangju_long.jpg',
        'gwangju_short.jpg'
    ]
    
    for alt_path in alternative_paths:
        image = cv2.imread(alt_path)
        if image is not None:
            print(f"대체 이미지 사용: {alt_path}")
            break
    
    if image is None:
        raise FileNotFoundError("사용할 수 있는 이미지 파일을 찾을 수 없습니다.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지에서 얼굴 감지 실행
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
    results = face_detection.process(image_rgb)
    
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)
        print(f"얼굴 {len(results.detections)}개 가 감지 되었습니다.")
    else:
        print("얼굴이 감지되지 않았습니다.")
        
# 결과 출력
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()