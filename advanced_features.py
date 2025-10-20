# 추가 기능 모듈들

import math
import json
import sqlite3
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances

class EmergencyAnalyzer:
    """긴급도 분석 클래스"""
    
    def __init__(self):
        self.emergency_keywords = {
            '매우긴급': 5,
            '긴급': 4,
            '싱크홀': 5,
            '대형': 4,
            '전기': 4,
            '가스': 5,
            '교통사고': 5,
            '응급': 5,
            '심각': 4,
            '위험': 4,
            '붕괴': 5,
            '화재': 5,
            '침수': 4,
            '교통마비': 4
        }
        
        self.damage_severity = {
            '가로등': {'기본': 2, '전기누전': 4, '전체소등': 3},
            '도로파손': {'기본': 3, '싱크홀': 5, '대형파손': 4, '교통마비': 4},
            '안전펜스': {'기본': 2, '완전파손': 3, '교통사고위험': 4},
            '불법주정차': {'기본': 1, '응급차량통과방해': 4, '교통마비': 3}
        }
    
    def analyze_emergency_level(self, damage_type: str, description: str = "", image_analysis: dict = None) -> int:
        """긴급도 분석"""
        base_urgency = 1
        
        # 기본 손상 유형별 긴급도
        if damage_type in self.damage_severity:
            base_urgency = self.damage_severity[damage_type].get('기본', 2)
        
        # 텍스트 분석
        text = f"{damage_type} {description}".lower()
        max_keyword_urgency = 0
        
        for keyword, urgency in self.emergency_keywords.items():
            if keyword in text:
                max_keyword_urgency = max(max_keyword_urgency, urgency)
        
        # 이미지 분석 결과 반영
        image_urgency = 0
        if image_analysis and 'detected_objects' in image_analysis:
            for obj in image_analysis['detected_objects']:
                if obj['label'].lower() in ['person', 'car', 'truck']:
                    image_urgency = max(image_urgency, 3)  # 사람이나 차량이 있으면 긴급도 상승
        
        # 최종 긴급도 계산
        final_urgency = max(base_urgency, max_keyword_urgency, image_urgency)
        
        return min(final_urgency, 5)  # 최대 5

class ClusterDetector:
    """군집 신고 탐지 클래스"""
    
    def __init__(self):
        self.cluster_threshold = 0.1  # 100m 반경
        self.min_cluster_size = 2  # 최소 군집 크기
    
    def detect_clusters(self, new_report: dict) -> List[dict]:
        """군집 신고 탐지"""
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        # 최근 24시간 내 신고 조회
        recent_time = datetime.now() - timedelta(hours=24)
        cursor.execute('''
            SELECT id, latitude, longitude, damage_type, created_at, urgency_level
            FROM reports 
            WHERE created_at > ? AND latitude IS NOT NULL AND longitude IS NOT NULL
        ''', (recent_time,))
        
        recent_reports = cursor.fetchall()
        conn.close()
        
        if len(recent_reports) < self.min_cluster_size:
            return []
        
        # 좌표 데이터 준비
        coordinates = []
        report_data = []
        
        for report in recent_reports:
            if report[1] and report[2]:  # latitude, longitude가 있는 경우
                coordinates.append([report[1], report[2]])
                report_data.append({
                    'id': report[0],
                    'damage_type': report[3],
                    'created_at': report[4],
                    'urgency_level': report[5]
                })
        
        if len(coordinates) < self.min_cluster_size:
            return []
        
        # DBSCAN 클러스터링
        try:
            # 위도/경도를 라디안으로 변환
            coords_rad = np.radians(coordinates)
            
            # 하버사인 거리 계산
            distances = haversine_distances(coords_rad) * 6371000  # 지구 반지름 (미터)
            
            # DBSCAN 적용
            clustering = DBSCAN(eps=self.cluster_threshold * 1000, min_samples=self.min_cluster_size, metric='precomputed')
            cluster_labels = clustering.fit_predict(distances)
            
            # 군집 정보 수집
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label != -1:  # 노이즈가 아닌 경우
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append({
                        'report_id': report_data[i]['id'],
                        'damage_type': report_data[i]['damage_type'],
                        'urgency_level': report_data[i]['urgency_level'],
                        'coordinates': coordinates[i]
                    })
            
            # 군집 정보 반환
            cluster_info = []
            for cluster_id, reports in clusters.items():
                if len(reports) >= self.min_cluster_size:
                    # 군집의 중심점 계산
                    center_lat = np.mean([r['coordinates'][0] for r in reports])
                    center_lon = np.mean([r['coordinates'][1] for r in reports])
                    
                    # 최고 긴급도
                    max_urgency = max([r['urgency_level'] for r in reports])
                    
                    cluster_info.append({
                        'cluster_id': cluster_id,
                        'report_count': len(reports),
                        'center_latitude': center_lat,
                        'center_longitude': center_lon,
                        'max_urgency': max_urgency,
                        'reports': reports
                    })
            
            return cluster_info
            
        except Exception as e:
            logger.error(f"군집 탐지 오류: {e}")
            return []

class ProcessingTimePredictor:
    """처리 시간 예측 클래스"""
    
    def __init__(self):
        self.base_times = {
            '가로등': {'평균': 24, '표준편차': 8},
            '도로파손': {'평균': 48, '표준편차': 16},
            '안전펜스': {'평균': 12, '표준편차': 4},
            '불법주정차': {'평균': 2, '표준편차': 1}
        }
        
        self.urgency_multipliers = {
            1: 1.5,
            2: 1.2,
            3: 1.0,
            4: 0.7,
            5: 0.3
        }
    
    def predict_processing_time(self, damage_type: str, urgency_level: int, cluster_info: List[dict] = None) -> str:
        """처리 시간 예측"""
        if damage_type not in self.base_times:
            damage_type = '기타'
            base_time = 24
            std_dev = 8
        else:
            base_time = self.base_times[damage_type]['평균']
            std_dev = self.base_times[damage_type]['표준편차']
        
        # 긴급도에 따른 시간 조정
        urgency_multiplier = self.urgency_multipliers.get(urgency_level, 1.0)
        
        # 군집 신고가 있는 경우 시간 단축
        cluster_bonus = 0.8 if cluster_info else 1.0
        
        # 예상 시간 계산 (정규분포 가정)
        estimated_hours = base_time * urgency_multiplier * cluster_bonus
        
        # 표준편차 적용 (95% 신뢰구간)
        confidence_interval = 1.96 * std_dev * urgency_multiplier * cluster_bonus
        
        min_hours = max(0.5, estimated_hours - confidence_interval)
        max_hours = estimated_hours + confidence_interval
        
        # 시간 포맷팅
        if max_hours < 1:
            return "30분 이내"
        elif max_hours < 24:
            return f"{int(min_hours)}-{int(max_hours)}시간 이내"
        else:
            min_days = int(min_hours / 24)
            max_days = int(max_hours / 24)
            return f"{min_days}-{max_days}일 이내"

class NotificationSystem:
    """알림 시스템"""
    
    def __init__(self):
        self.emergency_threshold = 4  # 긴급도 4 이상시 즉시 알림
    
    def should_send_emergency_notification(self, urgency_level: int, cluster_info: List[dict]) -> bool:
        """긴급 알림 발송 여부 판단"""
        if urgency_level >= self.emergency_threshold:
            return True
        
        if cluster_info and any(cluster['max_urgency'] >= self.emergency_threshold for cluster in cluster_info):
            return True
        
        return False
    
    def generate_notification_message(self, report_id: int, damage_type: str, urgency_level: int, cluster_info: List[dict]) -> str:
        """알림 메시지 생성"""
        urgency_text = ['낮음', '보통', '높음', '매우높음', '긴급'][urgency_level - 1]
        
        message = f"🚨 긴급 신고 알림\n\n"
        message += f"신고번호: #{report_id}\n"
        message += f"손상유형: {damage_type}\n"
        message += f"긴급도: {urgency_text}\n"
        
        if cluster_info:
            message += f"군집신고: {len(cluster_info)}개 군집 탐지\n"
            for cluster in cluster_info:
                message += f"- 군집 {cluster['cluster_id']}: {cluster['report_count']}건\n"
        
        message += f"\n즉시 현장 확인이 필요합니다."
        
        return message

# 전역 인스턴스 생성
emergency_analyzer = EmergencyAnalyzer()
cluster_detector = ClusterDetector()
time_predictor = ProcessingTimePredictor()
notification_system = NotificationSystem()
