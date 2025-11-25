from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pytesseract
import base64
import os
from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP
from sqlalchemy.orm import sessionmaker, declarative_base

# ================================
# CORS (React 연결)
# ================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# MySQL 연결
# ================================
DB_URL = "mysql+pymysql://root:1234@localhost/ai_vision"   # 비번 너꺼로 바꿔야 함

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ================================
# DB 모델 (license_plates table)
# ================================
class LicensePlate(Base):
    __tablename__ = "license_plates"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    plate_text = Column(String(100))
    created_at = Column(TIMESTAMP)

Base.metadata.create_all(bind=engine)

# ================================
# 번호판 OCR 함수들
# ================================
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 200)
    return edge

def find_plate_contour(edge):
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        approx = cv2.approxPolyDP(c, 10, True)
        if len(approx) == 4:
            return approx
    return None

def ocr_plate(img):
    config = "-l kor+eng --oem 3 --psm 7"
    return pytesseract.image_to_string(img, config=config).strip()

def encode_base64(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()

# ================================
# 업로드 + 분석 API
# ================================
@app.post("/upload-plate/")
async def upload_plate(file: UploadFile = File(...)):
    contents = await file.read()

    # 저장
    save_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(contents)

    # 이미지 읽기
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 전처리 → contour → 번호판 crop
    edge = preprocess(image)
    contour = find_plate_contour(edge)

    if contour is None:
        return {"error": "번호판을 찾지 못했습니다."}

    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    ys, xs = np.where(mask == 255)
    crop = image[min(ys):max(ys), min(xs):max(xs)]

    # OCR
    text = ocr_plate(crop)

    # DB 저장
    db = SessionLocal()
    new_record = LicensePlate(
        filename=file.filename,
        plate_text=text
    )
    db.add(new_record)
    db.commit()
    db.close()

    return {
        "filename": file.filename,
        "plate_text": text,
        "plate_image": encode_base64(crop)
    }
