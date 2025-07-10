# 얼굴 탐지 서비스 만들기
import cv2
import mediapipe as mp

# 미디어파이프 설정하기
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 이미지 불러오기
image_path = 'images/gwangju_long.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"{image_path} 이미지를 찾을 수 없습니다.")

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