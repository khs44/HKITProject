import cv2
from ultralytics import YOLO
import os

# 경로 설정
MODEL_PATH = "runs/detect/train/weights/best.pt"  # YOLO 번호판 모델 경로
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
VIDEO_PATH = "input.mp4"  # 원본 영상 경로
OUTPUT_PATH = "output.mp4"

# YOLOv8 번호판 탐지 모델 불러오기
yolo_model = YOLO(MODEL_PATH)

# Haar 얼굴 인식 모델 불러오기
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# 비디오 읽기
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 저장용 비디오 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ======================
    # YOLO 번호판 탐지
    # ======================
    results = yolo_model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = int(box.cls[0])
        if yolo_model.names[label] == 'license-plate':
            plate = frame[y1:y2, x1:x2]
            blurred_plate = cv2.GaussianBlur(plate, (23, 23), 30)
            frame[y1:y2, x1:x2] = blurred_plate

    # ======================
    # Haar 얼굴 탐지
    # ======================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (23, 23), 30)
        frame[y:y+h, x:x+w] = blurred_face

    # 비디오 저장
    out.write(frame)

    # 화면에도 보기
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
out.release()
cv2.destroyAllWindows()
