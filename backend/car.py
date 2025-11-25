#필요한 라이브러리 설치
#!pip install pytesseract
#!sudo apt-get install tesseract-ocr
#!sudo apt-get install tesseract-ocr-kor

# 필요한 라이브러리 임포트
import cv2
import numpy as np
import pytesseract
import base64
import os

# =============================================
# 1) 번호판 후보 영역 추출 (Edge + Contour 활용)
# =============================================
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)
    return edged

# =============================================
# 2) 번호판 영역(4각형) 찾기
# =============================================
def find_plate_contour(edged):
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    plate_contour = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)

        # 번호판은 보통 사각형(4개의 꼭짓점)
        if len(approx) == 4:
            plate_contour = approx
            break

    return plate_contour

# =============================================
# 3) OCR (번호판 문자 인식)
# =============================================
def recognize_plate_text(image):
    # 한국 차량 번호판 OCR
    config = "-l kor+eng --oem 3 --psm 7"
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()

# =============================================
# 4) base64 변환 (서버용)
# =============================================
def encode_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

# =============================================
# 5) 전체 처리 함수 (이미지 → 번호판 crop → 문자 인식)
# =============================================
def detect_license_plate(image_path):

    if not os.path.exists(image_path):
        return {"error": "이미지가 존재하지 않습니다."}

    image = cv2.imread(image_path)

    edged = preprocess_image(image)
    plate_contour = find_plate_contour(edged)

    if plate_contour is None:
        return {"error": "번호판 윤곽을 찾지 못했습니다."}

    # 번호판 영역 마스크 처리
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [plate_contour], -1, 255, -1)

    plate_img = cv2.bitwise_and(image, image, mask=mask)

    # 번호판 crop 영역 계산
    y_indices, x_indices = np.where(mask == 255)
    top_y, top_x = np.min(y_indices), np.min(x_indices)
    bottom_y, bottom_x = np.max(y_indices), np.max(x_indices)

    cropped_plate = image[top_y:bottom_y + 1, top_x:bottom_x + 1]

    # OCR 문자 인식
    plate_text = recognize_plate_text(cropped_plate)

    return {
        "plate_text": plate_text,
        "plate_image_base64": encode_base64(cropped_plate),
    }

# =============================================
# 6) 로컬 테스트 실행
# python license_plate_detector.py
# =============================================
if __name__ == "__main__":

    test_images = [
        "car1.jpg",
        "car2.jpg",
    ]
    test_mp4s = [
        "car1.jpg",
        "car2.jpg",
    ]

    for img in test_images:
        print(f"▶ {img} 분석 중...")
        result = detect_license_plate(img)
        print(result)
        print("------------------------------------------")
