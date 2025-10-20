# 공공기물 파손 신고 시스템

## 프로젝트 개요
시민이 웹 인터페이스를 통해 공공기물 파손을 쉽게 신고할 수 있는 AI 기반 스마트 신고 시스템

### 해결하고자 하는 문제
- 가로등 파손, 도로 파손, 도로 안전 펜스 파손 시 신고의 번거로움
- 기존 신고 시스템의 낮은 사용성
- 민원센터 전화 신고의 불편함

### 개선 방안
- 웹 기반 채팅형 인터페이스를 통한 간편한 신고
- AI 객체 탐지로 자동 손상 유형 분류
- 위치 정보 자동 추출
- 부서별 자동 연계
- 실시간 처리 상태 업데이트

## 주요 기능

### AI 기반 기능
- **이미지 객체 탐지**: Hugging Face DETR/YOLO 모델 사용 (선택적)
- **손상 유형 자동 분류**: 가로등, 도로파손, 안전펜스, 불법주정차 등
- **긴급도 AI 판단**: 텍스트 키워드 분석 기반 긴급도 계산 (1-5단계)
- **군집 신고 탐지**: DBSCAN 알고리즘으로 동일 지역 다중 신고 탐지

### 사용자 기능
- **채팅형 UI**: 카카오톡과 유사한 직관적인 인터페이스
- **사진 업로드**: 드래그 앤 드롭 또는 클릭으로 간편 업로드
- **실시간 상태 확인**: 신고 처리 과정 실시간 모니터링
- **손상 유형 선택**: AI 추천 + 사용자 선택 옵션
- **지도 기능**: 위치 확인 및 신고 현황 시각화 (선택적)
- **EXIF 위치 추출**: 사진에서 GPS 정보 자동 추출

### 관리자 기능
- **실시간 대시보드**: 신고 현황 및 통계 모니터링
- **군집 신고 관리**: 동일 지역 다중 신고 우선 처리
- **긴급 알림 시스템**: 긴급도 4-5단계 신고 즉시 알림

## 기술 스택

### Backend
- **FastAPI**: 고성능 비동기 웹 프레임워크
- **Python 3.8+**: 메인 개발 언어
- **SQLite**: 경량 데이터베이스

### AI/ML (선택적)
- **Hugging Face Transformers**: 객체 탐지 모델 (DETR/YOLO)
- **OpenCV**: 이미지 처리
- **scikit-learn**: 군집 분석 (DBSCAN)
- **PyTorch**: 딥러닝 프레임워크
- **PIL**: 이미지 처리 및 EXIF 데이터 추출

### Frontend
- **HTML5/CSS3/JavaScript**: 반응형 웹 UI
- **카카오톡 스타일 디자인**: 친숙한 사용자 경험
- **Leaflet 지도**: 위치 시각화 (선택적)

## 프로젝트 구조

```
vandalism/
├── main.py                 # 고급 기능 포함 메인 서버
├── simple_main.py          # 기본 기능만 포함 간단 버전
├── advanced_features.py    # AI 고급 기능 모듈
├── requirements.txt        # Python 의존성
├── run.sh                 # 실행 스크립트
├── reports.db             # SQLite 데이터베이스
├── static/                # 정적 파일
│   ├── index.html         # 기본 사용자 인터페이스
│   ├── index_with_map.html # 지도 기능 포함 인터페이스
│   └── admin.html         # 관리자 대시보드
├── uploads/               # 업로드된 이미지 저장
└── venv/                  # 가상환경
```

## 설치 및 실행

### 1. 저장소 클론
```bash
git clone <repository-url>
cd vandalism
```

### 2. 가상환경 설정
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 서버 실행
```bash
# 방법 1: 실행 스크립트 사용
./run.sh

# 방법 2: 직접 실행 (기본 버전)
uvicorn simple_main:app --reload --host 0.0.0.0 --port 8000

# 방법 3: 고급 기능 포함 버전
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 접속
- **사용자 인터페이스**: http://localhost:8000
- **관리자 대시보드**: http://localhost:8000/admin
- **API 문서**: http://localhost:8000/docs

## API 엔드포인트

### 사용자 API
- `POST /api/upload` - 이미지 업로드 및 AI 분석
- `POST /api/report` - 신고 접수
- `GET /api/report/{report_id}` - 신고 상태 조회
- `GET /api/damage-types` - 손상 유형 목록

### 관리자 API
- `GET /api/reports` - 신고 목록 조회
- `GET /api/statistics` - 신고 통계
- `GET /api/clusters` - 군집 신고 현황
- `POST /api/report/{report_id}/status` - 신고 상태 업데이트

## 차별화 포인트

### 1. AI 기반 자동화
- **객체 탐지**: 사진만으로 손상 유형 자동 분류 (선택적)
- **긴급도 판단**: 텍스트 키워드 분석으로 정확한 긴급도 계산
- **군집 탐지**: 동일 지역 다중 신고 자동 탐지 및 우선 처리

### 2. 사용자 경험 최적화
- **카카오톡 스타일 UI**: 친숙한 채팅 인터페이스
- **간편한 신고**: 사진 업로드 → AI 분석 → 자동 신고
- **실시간 피드백**: 처리 과정 실시간 업데이트

### 3. 관리 효율성
- **스마트 대시보드**: 실시간 현황 모니터링
- **자동 우선순위**: 긴급도 및 군집 기반 자동 우선순위 설정
- **예측 분석**: 처리 시간 예측 및 리소스 최적화

## 시스템 아키텍처

```
사용자 → 웹 인터페이스 → FastAPI 서버 → AI 분석 엔진
                                    ↓
관리자 대시보드 ← SQLite DB ← 부서 연계 시스템
```
