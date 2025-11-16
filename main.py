from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import sqlite3
from datetime import datetime
import json
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from transformers import pipeline
import requests
import logging
import sqlite3
from datetime import timedelta
import json
from advanced_features import (
    emergency_analyzer, cluster_detector, time_predictor, notification_system
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="공공기물 파손 신고 챗봇", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 데이터베이스 초기화
def init_db():
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            image_path TEXT,
            location TEXT,
            latitude REAL,
            longitude REAL,
            damage_type TEXT,
            urgency_level INTEGER,
            description TEXT,
            status TEXT DEFAULT '접수',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            damage_type TEXT,
            department_name TEXT,
            contact_info TEXT
        )
    ''')
    
    # 부서 정보 초기 데이터
    departments = [
        ('가로등', '시설관리과', '02-1234-5678'),
        ('도로파손', '도로관리과', '02-1234-5679'),
        ('안전펜스', '교통안전과', '02-1234-5680'),
        ('불법주정차', '교통단속과', '02-1234-5681')
    ]
    
    cursor.execute('DELETE FROM departments')
    cursor.executemany('INSERT INTO departments (damage_type, department_name, contact_info) VALUES (?, ?, ?)', departments)
    
    conn.commit()
    conn.close()

# Pydantic 모델들
class ReportRequest(BaseModel):
    user_id: str
    description: Optional[str] = None
    damage_type: Optional[str] = None

class ReportResponse(BaseModel):
    report_id: int
    status: str
    message: str
    damage_type: str
    urgency_level: int
    department: str
    estimated_time: str

class ReportStatus(BaseModel):
    report_id: int
    status: str
    created_at: str
    updated_at: str
    department: str
    progress: str

# AI 모델 초기화
object_detector = None
try:
    # 더 정확한 객체 탐지 모델 사용 (YOLOv8 기반)
    logger.info("AI 모델을 로딩하는 중...")
    # YOLOv8 모델 사용 (더 정확한 객체 탐지)
    object_detector = pipeline("object-detection", model="hustvl/yolos-tiny", device=-1)    # hustvl/yolos-tiny
    logger.info("✅ AI 모델 로드 완료!")
except Exception as e:
    logger.error(f"❌ YOLOv8 모델 로드 실패: {e}")
    try:
        # 대안 모델 시도
        logger.info("대안 모델 로딩 시도...")
        object_detector = pipeline("object-detection", model="facebook/detr-resnet-50", device=-1)    #acebook/detr-resnet-50
        logger.info("✅ 대안 AI 모델 로드 완료!")
    except Exception as e2:
        logger.error(f"❌ 모든 AI 모델 로드 실패: {e2}")
        logger.info("⚠️ AI 기능 없이 기본 모드로 실행합니다.")
        object_detector = None

# 객체 라벨 한글 번역 함수
def translate_object_label(english_label: str) -> str:
    """영어 객체 라벨을 한글로 번역"""
    translation_map = {
        # 차량 관련
        'car': '자동차',
        'truck': '트럭',
        'bus': '버스',
        'motorcycle': '오토바이',
        'bicycle': '자전거',
        'vehicle': '차량',
        'illegal_parking': '차량',
        
        # 가로등 관련
        'traffic light': '신호등',
        'pole': '전봇대',
        'lamp': '가로등',
        'street light': '가로등',
        'street_light': '가로등',
        'streetlight': '가로등',
        
        # 도로 관련
        'road': '도로',
        'road_damage': '도로',
        'street': '도로',
        'highway': '고속도로',
        'pavement': '포장도로',
        'asphalt': '아스팔트',
        'concrete': '콘크리트',
        
        # 안전 시설
        'fence': '펜스',
        'safety_fence': '펜스',
        'barrier': '방호벽',
        'guardrail': '가드레일',
        'railing': '난간',
        
        # 기타 공공시설
        'person': '사람',
        'stop sign': '정지표지판',
        'fire hydrant': '소화전',
        'bench': '벤치',
        'sign': '표지판',
        'building': '건물',
        'tree': '나무',
        'house': '집',
        'window': '창문',
        'door': '문',
        'chair': '의자',
        'table': '테이블',
        'bottle': '병',
        'cup': '컵',
        'book': '책',
        'laptop': '노트북',
        'keyboard': '키보드',
        'mouse': '마우스',
        'tv': 'TV',
        'remote': '리모컨',
        'scissors': '가위'
    }
    
    return translation_map.get(english_label.lower(), english_label)

# 긴급도 판단 함수 (고급 기능 사용)
def calculate_urgency(damage_type: str, description: str = "", image_analysis: dict = None) -> int:
    """긴급도 계산 (1-5, 5가 가장 긴급)"""
    return emergency_analyzer.analyze_emergency_level(damage_type, description, image_analysis)

# 이미지 분석 함수
def analyze_image(image_bytes: bytes) -> dict:
    """이미지에서 객체 탐지 및 분석"""
    if not object_detector:
        # AI 모델이 없을 때 기본 분석
        try:
            image = Image.open(BytesIO(image_bytes))
            return {
                "damage_type": "기타",
                "confidence": 0.1,
                "detected_objects": [],
                "analysis": "이미지가 성공적으로 업로드되었습니다. 손상 유형을 선택해주세요.",
                "ai_enabled": False
            }
        except Exception as e:
            return {"error": f"이미지 로드 실패: {str(e)}"}
    
    try:
        # 이미지 로드
        image = Image.open(BytesIO(image_bytes))
        
        # 객체 탐지
        results = object_detector(image)
        
        # 탐지할 객체 개수 설정
        # 테스트용: 1개, 3개, 5개 등으로 변경해서 테스트 가능
        MAX_OBJECTS = 5
        
        # 설정된 개수만큼 객체 표시 - 한글 번역 적용
        # detected_objects = []
        # for i, result in enumerate(results[:MAX_OBJECTS]):
        #     detected_objects.append({
        #         'label': translate_object_label(result['label']),  # 한글 번역
        #         'score': result['score'],
        #         'box': result['box'],
        #         'original_label': result['label']  # 원본 영어 라벨 보존
        #     })

        detected_objects = []
        yolo_result = results[0]
        
        for box in yolo_result.boxes[:MAX_OBJECTS]:
            cls = int(box.cls[0])
            score = float(box.conf[0])
            label = object_detector.names[cls]
        
            detected_objects.append({
                "label": translate_object_label(label),
                "score": score,
                "box": box.xyxy.tolist(),
                "original_label": label
            })
        
        # 공공기물 관련 객체 필터링 및 손상 유형 추정
        public_objects = [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 
            'traffic light', 'stop sign', 'fire hydrant', 'bench',
            'pole', 'lamp', 'street light', 'road', 'street', 'highway',
            'fence', 'barrier', 'guardrail', 'railing', 'sign', 'building'
        ]
        
        # 객체에 따른 손상 유형 매핑
        object_to_damage = {
            'car': '불법주정차',
            'truck': '불법주정차', 
            'bus': '불법주정차',
            'motorcycle': '불법주정차',
            'bicycle': '불법주정차',
            'illegal_parking': '불법주정차',
            'person': '기타',
            'traffic light': '가로등',
            'stop sign': '기타',
            'fire hydrant': '기타',
            'bench': '기타',
            'pole': '가로등',
            'lamp': '가로등',
            'street light': '가로등',
            'street_light': '가로등',
            'road': '도로',
            'street': '도로',
            'highway': '도로',
            'road': '도로',
            'damage_road': '도로',
            'fence': '안전펜스',
            'safety_fence': '안전펜스',
            'barrier': '안전펜스',
            'guardrail': '안전펜스',
            'railing': '안전펜스',
            'sign': '기타',
            'building': '기타'
        }
        
        # 탐지된 객체들로 손상 유형 결정
        if detected_objects:
            # 공공기물이 있는지 확인
            public_detected = []
            for obj in detected_objects:
                if obj['original_label'].lower() in public_objects:
                    public_detected.append(obj)
            
            if public_detected:
                # 공공기물이 있으면 가장 높은 신뢰도의 공공기물 사용
                best_object = max(public_detected, key=lambda x: x['score'])
                damage_type = object_to_damage.get(best_object['original_label'].lower(), '기타')
                confidence = best_object['score']
            else:
                # 공공기물이 없으면 가장 높은 신뢰도의 객체 사용
                best_object = detected_objects[0]
                damage_type = "기타"
                confidence = best_object['score']
            
            # 분석 메시지 생성 (여러 객체인 경우)
            if len(detected_objects) == 1:
                analysis = f"탐지된 객체: {detected_objects[0]['label']}"
            else:
                object_names = [obj['label'] for obj in detected_objects]
                analysis = f"탐지된 객체: {', '.join(object_names)}"
        else:
            damage_type = "기타"
            confidence = 0.0
            analysis = "탐지된 객체가 없습니다. 수동으로 손상 유형을 선택해주세요."
        
        return {
            "damage_type": damage_type,
            "confidence": confidence,
            "detected_objects": detected_objects,
            "analysis": analysis,
            "ai_enabled": True
        }
        
    except Exception as e:
        logger.error(f"이미지 분석 오류: {e}")
        return {"error": f"이미지 분석 실패: {str(e)}"}

# 위치 정보 추출 (EXIF 데이터에서)
def extract_location(image_bytes: bytes) -> dict:
    """이미지에서 위치 정보 추출"""
    try:
        image = Image.open(BytesIO(image_bytes))
        
        # EXIF 데이터 안전하게 처리
        try:
            exif = image._getexif()
            if exif:
                # GPS 정보 추출 (실제 구현에서는 더 정교한 GPS 파싱 필요)
                gps_info = exif.get(34853)  # GPS 태그
                if gps_info:
                    # GPS 좌표 처리 (tuple 형태일 수 있음)
                    try:
                        lat_raw = gps_info.get(2)
                        lon_raw = gps_info.get(4)
                        
                        # tuple인 경우 도분초를 도로 변환
                        if isinstance(lat_raw, tuple) and len(lat_raw) >= 3:
                            lat = lat_raw[0] + lat_raw[1]/60.0 + lat_raw[2]/3600.0
                        elif isinstance(lat_raw, (int, float)):
                            lat = float(lat_raw)
                        else:
                            lat = 0
                            
                        if isinstance(lon_raw, tuple) and len(lon_raw) >= 3:
                            lon = lon_raw[0] + lon_raw[1]/60.0 + lon_raw[2]/3600.0
                        elif isinstance(lon_raw, (int, float)):
                            lon = float(lon_raw)
                        else:
                            lon = 0
                            
                        # 주소 변환
                        address = reverse_geocode(lat, lon)
                        return {
                            "latitude": lat,
                            "longitude": lon,
                            "location": address
                        }
                    except Exception as coord_error:
                        logger.warning(f"GPS 좌표 변환 오류 (무시됨): {coord_error}")
        except Exception as exif_error:
            logger.warning(f"EXIF 데이터 처리 오류 (무시됨): {exif_error}")
        
        # GPS 정보가 없으면 기본값
        return {
            "latitude": 37.5665,  # 서울시청 좌표
            "longitude": 126.9780,
            "location": "서울특별시 중구 세종대로 110"
        }
        
    except Exception as e:
        logger.error(f"위치 정보 추출 오류: {e}")
        return {
            "latitude": 37.5665,
            "longitude": 126.9780, 
            "location": "서울특별시 중구 세종대로 110"
        }

# 위도/경도를 주소로 변환하는 함수 (실제 API 사용)
def reverse_geocode(latitude: float, longitude: float) -> str:
    """위도/경도를 주소로 변환 (Nominatim API 사용)"""
    try:
        # Nominatim (OpenStreetMap) 무료 지오코딩 API 사용
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": latitude,
            "lon": longitude,
            "format": "json",
            "addressdetails": 1,
            "accept-language": "ko"
        }
        headers = {
            "User-Agent": "PublicDamageReportBot/1.0"  # API 사용을 위한 User-Agent
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("address"):
                address = data["address"]
                
                # 한국 주소 형식으로 구성
                country = address.get("country", "")
                state = address.get("state", "")
                city = address.get("city", "")
                town = address.get("town", "")
                village = address.get("village", "")
                suburb = address.get("suburb", "")
                neighbourhood = address.get("neighbourhood", "")
                
                # 주소 구성
                if country == "대한민국" or country == "South Korea":
                    if state and city:
                        if town:
                            return f"{state} {city} {town}"
                        elif village:
                            return f"{state} {city} {village}"
                        elif suburb:
                            return f"{state} {city} {suburb}"
                        else:
                            return f"{state} {city}"
                    elif state:
                        return state
                
                # 기본 주소 반환
                display_name = data.get("display_name", "")
                if display_name:
                    # 한국어 주소 부분만 추출
                    parts = display_name.split(", ")
                    korean_parts = [part for part in parts if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in part)]
                    if korean_parts:
                        return korean_parts[0]
        
        # API 실패 시 좌표 반환
        return f"위도: {latitude:.6f}, 경도: {longitude:.6f}"
        
    except Exception as e:
        logger.warning(f"주소 변환 오류: {e}")
        return f"위도: {latitude:.6f}, 경도: {longitude:.6f}"

# 부서 연계 함수
def get_department(damage_type: str) -> dict:
    """손상 유형에 따른 담당 부서 조회"""
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT department_name, contact_info FROM departments WHERE damage_type = ?', (damage_type,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return {
            "department": result[0],
            "contact": result[1]
        }
    else:
        return {
            "department": "시설관리과",
            "contact": "02-1234-5678"
        }

# 처리 예상 시간 계산 (고급 기능 사용)
def estimate_processing_time(damage_type: str, urgency_level: int, cluster_info: List[dict] = None) -> str:
    """처리 예상 시간 계산"""
    return time_predictor.predict_processing_time(damage_type, urgency_level, cluster_info)

@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("데이터베이스 초기화 완료")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """메인 페이지 (지도 기능 포함)"""
    with open("static/index_with_map.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """관리자 대시보드"""
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/reports")
async def get_reports(limit: int = 50, status: str = None):
    """신고 목록 조회 (관리자용)"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        query = '''
            SELECT id, user_id, damage_type, urgency_level, status, created_at, updated_at
            FROM reports
        '''
        params = []
        
        if status:
            query += ' WHERE status = ?'
            params.append(status)
        
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        reports = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": report[0],
                "user_id": report[1],
                "damage_type": report[2],
                "urgency_level": report[3],
                "status": report[4],
                "created_at": report[5],
                "updated_at": report[6]
            }
            for report in reports
        ]
        
    except Exception as e:
        logger.error(f"신고 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신고 목록 조회 실패: {str(e)}")

@app.post("/api/upload", response_model=dict)
async def upload_image(file: UploadFile = File(...)):
    """이미지 업로드 및 분석"""
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        
        # 이미지 분석
        analysis = analyze_image(image_bytes)
        
        # 위치 정보 추출
        location_info = extract_location(image_bytes)
        
        # 이미지 저장
        os.makedirs("uploads", exist_ok=True)
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = f"uploads/{filename}"
        
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        
        return {
            "success": True,
            "analysis": analysis,
            "location": location_info,
            "filename": filename,
            "message": "이미지 분석이 완료되었습니다."
        }
        
    except Exception as e:
        logger.error(f"이미지 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 업로드 실패: {str(e)}")

# 자동 상태 업데이트 함수
async def auto_update_status():
    """자동으로 신고 상태를 업데이트"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        # 현재 시간 기준으로 상태 업데이트
        cursor.execute('''
            UPDATE reports 
            SET status = CASE 
                WHEN status = '접수' AND datetime(created_at, '+1 hour') <= datetime('now') THEN '검토중'
                WHEN status = '검토중' AND datetime(created_at, '+4 hours') <= datetime('now') THEN '처리중'
                WHEN status = '처리중' AND datetime(created_at, '+8 hours') <= datetime('now') THEN '완료'
                ELSE status
            END,
            updated_at = CURRENT_TIMESTAMP
            WHERE status IN ('접수', '검토중', '처리중')
        ''')
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"자동 상태 업데이트 오류: {e}")

@app.post("/api/report", response_model=ReportResponse)
async def create_report(request: ReportRequest):
    """신고 접수"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        # 기본값 설정
        damage_type = request.damage_type or "기타"
        urgency_level = calculate_urgency(damage_type, request.description or "")
        
        # 부서 정보 조회
        dept_info = get_department(damage_type)
        
        # 신고 데이터 삽입
        cursor.execute('''
            INSERT INTO reports (user_id, damage_type, description, urgency_level, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (request.user_id, damage_type, request.description, urgency_level, '접수'))
        
        report_id = cursor.lastrowid
        
        # 위치 정보 업데이트 (이미지 분석에서 가져온 경우)
        if hasattr(request, 'latitude') and hasattr(request, 'longitude'):
            cursor.execute('''
                UPDATE reports SET latitude = ?, longitude = ? WHERE id = ?
            ''', (request.latitude, request.longitude, report_id))
        
        conn.commit()
        conn.close()
        
        # 군집 신고 탐지
        cluster_info = cluster_detector.detect_clusters({
            'damage_type': damage_type,
            'urgency_level': urgency_level
        })
        
        # 처리 예상 시간 계산 (군집 정보 포함)
        estimated_time = estimate_processing_time(damage_type, urgency_level, cluster_info)
        
        # 긴급 알림 발송 여부 확인
        should_notify = notification_system.should_send_emergency_notification(urgency_level, cluster_info)
        
        response_message = "신고가 성공적으로 접수되었습니다."
        if cluster_info:
            response_message += f" 동일 지역에서 {len(cluster_info)}개의 군집 신고가 탐지되어 우선 처리됩니다."
        
        if should_notify:
            response_message += " 긴급 신고로 분류되어 즉시 알림이 발송됩니다."
            logger.info(f"긴급 알림 발송: 신고 #{report_id}")
        
        return ReportResponse(
            report_id=report_id,
            status="접수",
            message=response_message,
            damage_type=damage_type,
            urgency_level=urgency_level,
            department=dept_info["department"],
            estimated_time=estimated_time
        )
        
    except Exception as e:
        logger.error(f"신고 접수 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신고 접수 실패: {str(e)}")

@app.get("/api/report/{report_id}", response_model=ReportStatus)
async def get_report_status(report_id: int):
    """신고 상태 조회"""
    try:
        # 자동 상태 업데이트 실행
        await auto_update_status()
        
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT status, created_at, updated_at, damage_type
            FROM reports WHERE id = ?
        ''', (report_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="신고를 찾을 수 없습니다.")
        
        status, created_at, updated_at, damage_type = result
        dept_info = get_department(damage_type)
        
        # 진행 상황 매핑
        progress_map = {
            '접수': '신고 접수 완료',
            '검토중': '담당 부서 검토 중',
            '처리중': '현장 조사 및 처리 중',
            '완료': '처리 완료'
        }
        
        return ReportStatus(
            report_id=report_id,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            department=dept_info["department"],
            progress=progress_map.get(status, "처리 중")
        )
        
    except Exception as e:
        logger.error(f"신고 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신고 상태 조회 실패: {str(e)}")

@app.get("/api/damage-types")
async def get_damage_types():
    """손상 유형 목록 조회"""
    return {
        "damage_types": [
            {"id": "가로등", "name": "가로등", "options": ["가로등 꺼짐", "가로등 부서짐", "가로등 깜빡임"]},
            {"id": "도로파손", "name": "도로 파손", "options": ["포장 파손", "싱크홀", "도로 함몰", "차선 불분명"]},
            {"id": "안전펜스", "name": "안전 펜스", "options": ["펜스 파손", "펜스 누락", "펜스 기울어짐"]},
            {"id": "불법주정차", "name": "불법 주정차", "options": ["일반 주정차", "장기 주정차", "위험 주정차"]},
            {"id": "기타", "name": "기타", "options": ["기타 시설물", "기타 안전사고"]}
        ]
    }

@app.get("/api/clusters")
async def get_cluster_reports():
    """군집 신고 현황 조회"""
    try:
        # 최근 7일간의 군집 신고 조회
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        recent_time = datetime.now() - timedelta(days=7)
        cursor.execute('''
            SELECT id, damage_type, latitude, longitude, urgency_level, created_at
            FROM reports 
            WHERE created_at > ? AND latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY created_at DESC
        ''', (recent_time,))
        
        reports = cursor.fetchall()
        conn.close()
        
        if len(reports) < 2:
            return {"clusters": [], "message": "군집 신고가 없습니다."}
        
        # 군집 탐지
        cluster_info = cluster_detector.detect_clusters({})
        
        return {
            "clusters": cluster_info,
            "total_reports": len(reports),
            "cluster_count": len(cluster_info)
        }
        
    except Exception as e:
        logger.error(f"군집 신고 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"군집 신고 조회 실패: {str(e)}")

@app.get("/api/statistics")
async def get_statistics():
    """신고 통계 조회"""
    try:
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        # 전체 신고 수
        cursor.execute('SELECT COUNT(*) FROM reports')
        total_reports = cursor.fetchone()[0]
        
        # 긴급도별 신고 수
        cursor.execute('''
            SELECT urgency_level, COUNT(*) 
            FROM reports 
            GROUP BY urgency_level 
            ORDER BY urgency_level
        ''')
        urgency_stats = dict(cursor.fetchall())
        
        # 손상 유형별 신고 수
        cursor.execute('''
            SELECT damage_type, COUNT(*) 
            FROM reports 
            GROUP BY damage_type 
            ORDER BY COUNT(*) DESC
        ''')
        damage_type_stats = dict(cursor.fetchall())
        
        # 최근 24시간 신고 수
        recent_time = datetime.now() - timedelta(hours=24)
        cursor.execute('SELECT COUNT(*) FROM reports WHERE created_at > ?', (recent_time,))
        recent_reports = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_reports": total_reports,
            "recent_24h": recent_reports,
            "urgency_distribution": urgency_stats,
            "damage_type_distribution": damage_type_stats,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

@app.post("/api/report/{report_id}/status")
async def update_report_status(report_id: int, status: str):
    """신고 상태 업데이트 (관리자용)"""
    try:
        valid_statuses = ['접수', '검토중', '처리중', '완료']
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 상태입니다. 가능한 상태: {valid_statuses}")
        
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE reports 
            SET status = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (status, report_id))
        
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="신고를 찾을 수 없습니다.")
        
        conn.commit()
        conn.close()
        
        return {"message": f"신고 #{report_id}의 상태가 '{status}'로 업데이트되었습니다."}
        
    except Exception as e:
        logger.error(f"신고 상태 업데이트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"신고 상태 업데이트 실패: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
