import cv2
from ultralytics import YOLO

# 모델 불러오기 (너가 학습한 커스텀 모델 사용)
model = YOLO("runs/detect/train/weights/best.pt")  # 모델 경로 수정

# 비디오 파일 열기 (또는 0이면 웹캠)
cap = cv2.VideoCapture("video/car_video.mp4")  # 경로 수정

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 탐지
    results = model(frame)[0]

    # 탐지된 박스에 대해 반복
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = int(box.cls[0])

        # 번호판으로 분류된 객체만 처리 (클래스 이름이 'license-plate'일 때만)
        if model.names[label] == 'license-plate':
            roi = frame[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (23, 23), 30)
            frame[y1:y2, x1:x2] = blurred

    # 결과 출력
    cv2.imshow("Blurred License Plate", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
