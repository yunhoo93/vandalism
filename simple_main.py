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
import logging

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

# 긴급도 판단 함수 (간단한 버전)
def calculate_urgency(damage_type: str, description: str = "") -> int:
    """긴급도 계산 (1-5, 5가 가장 긴급)"""
    urgency_keywords = {
        '싱크홀': 5,
        '대형': 4,
        '전기': 4,
        '가스': 5,
        '교통사고': 5,
        '응급': 5,
        '긴급': 4,
        '심각': 4,
        '위험': 4
    }
    
    text = f"{damage_type} {description}".lower()
    
    for keyword, urgency in urgency_keywords.items():
        if keyword in text:
            return urgency
    
    # 기본 긴급도
    if damage_type in ['가로등', '도로파손']:
        return 3
    elif damage_type in ['안전펜스', '불법주정차']:
        return 2
    else:
        return 1

# 간단한 이미지 분석 함수 (AI 없이)
def analyze_image_simple(image_bytes: bytes) -> dict:
    """간단한 이미지 분석 (AI 없이)"""
    try:
        # 이미지 로드
        image = Image.open(BytesIO(image_bytes))
        
        # 기본 정보만 반환
        return {
            "damage_type": "기타",
            "confidence": 0.5,
            "detected_objects": [],
            "analysis": "이미지가 성공적으로 업로드되었습니다. 손상 유형을 선택해주세요."
        }
        
    except Exception as e:
        logger.error(f"이미지 분석 오류: {e}")
        return {"error": f"이미지 분석 실패: {str(e)}"}

# 위치 정보 추출 (기본값)
def extract_location(image_bytes: bytes) -> dict:
    """기본 위치 정보 반환"""
    return {
        "latitude": 37.5665,  # 서울시청 좌표
        "longitude": 126.9780,
        "location": "서울특별시 중구 세종대로 110"
    }

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

# 처리 예상 시간 계산
def estimate_processing_time(damage_type: str, urgency_level: int) -> str:
    """처리 예상 시간 계산"""
    base_times = {
        '가로등': 24,
        '도로파손': 48,
        '안전펜스': 12,
        '불법주정차': 2
    }
    
    base_time = base_times.get(damage_type, 24)
    
    # 긴급도에 따른 시간 조정
    urgency_multiplier = {
        1: 1.5,
        2: 1.2,
        3: 1.0,
        4: 0.7,
        5: 0.3
    }
    
    estimated_hours = base_time * urgency_multiplier.get(urgency_level, 1.0)
    
    if estimated_hours < 1:
        return "30분 이내"
    elif estimated_hours < 24:
        return f"{int(estimated_hours)}시간 이내"
    else:
        return f"{int(estimated_hours/24)}일 이내"

@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("데이터베이스 초기화 완료")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """메인 페이지"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """관리자 대시보드"""
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/upload", response_model=dict)
async def upload_image(file: UploadFile = File(...)):
    """이미지 업로드 및 분석"""
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        
        # 간단한 이미지 분석
        analysis = analyze_image_simple(image_bytes)
        
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
            "message": "이미지가 성공적으로 업로드되었습니다."
        }
        
    except Exception as e:
        logger.error(f"이미지 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 업로드 실패: {str(e)}")

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
        
        conn.commit()
        conn.close()
        
        # 처리 예상 시간 계산
        estimated_time = estimate_processing_time(damage_type, urgency_level)
        
        return ReportResponse(
            report_id=report_id,
            status="접수",
            message="신고가 성공적으로 접수되었습니다.",
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
