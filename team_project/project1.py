import cv2
import numpy as np
import torch
from ultralytics import YOLO
from roboflow import Roboflow

# Roboflow API 설정
rf = Roboflow(api_key="tEFOW2xP2wtvNyZmpXnL")
project = rf.workspace("jyk-ucnhk").project("jyk")
version = project.version(11)
dataset = version.download("yolov11")

# YOLO11n 모델 로드
model = YOLO('yolo11n.pt')  # YOLO11n 모델 로드

# YOLO11n 모델 설정
CONFIDENCE_THRESHOLD = 0.25  # 신뢰도 임계값
IOU_THRESHOLD = 0.45        # IOU 임계값
MAX_DETECTIONS = 100        # 최대 감지 객체 수

# YOLO11n 클래스 정의
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light']

# 클래스별 색상 정의
COLORS = {
    'person': (255, 0, 0),      # 파란색
    'vehicle': (0, 255, 0),     # 녹색 (차량 관련)
    'traffic light': (0, 255, 255)  # 청록색
}

# 클래스 카테고리 정의
VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle']

def get_class_color(class_name):
    if class_name == 'person':
        return COLORS['person']
    elif class_name in VEHICLE_CLASSES:
        return COLORS['vehicle']
    elif class_name == 'traffic light':
        return COLORS['traffic light']
    else:
        return (0, 0, 255)  # 빨간색 (기본)

# 마우스 이벤트 관련 변수
drawing = False
points = []  # 다각형의 꼭지점을 저장할 리스트
roi_list = []  # 여러 ROI를 저장할 리스트
current_roi = None

def draw_polygon(event, x, y, flags, param):
    global drawing, points, current_roi
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 왼쪽 클릭: 꼭지점 추가
        points.append((x, y))
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 오른쪽 클릭: 현재 다각형 완성
        if len(points) >= 3:  # 최소 3개의 점이 필요
            roi_list.append(points.copy())
        points = []  # 다음 다각형을 위해 초기화
        
    elif event == cv2.EVENT_MBUTTONDOWN:
        # 가운데 버튼 클릭: 마지막 ROI 삭제
        if roi_list:
            roi_list.pop()

def is_point_in_polygon(point, polygon):
    # 점이 다각형 내부에 있는지 확인
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

# 영상 캡처
cap = cv2.VideoCapture("video\KakaoTalk_20250414_114947758.mp4")
cv2.namedWindow('Object Detection')
cv2.setMouseCallback('Object Detection', draw_polygon)

# 객체 카운트를 위한 딕셔너리
object_counts = {class_name: 0 for class_name in CLASSES}

# 객체의 이전 위치와 방향을 저장할 딕셔너리
previous_positions = {}
movement_directions = {}
movement_status = {}

# FPS를 저장하는 변수 (초당 100프레임)
TARGET_FPS = 100
# 프레임 간 시간 간격 (1/100초)
FRAME_INTERVAL = 1.0 / TARGET_FPS

# 픽셀과 실제 거리 간의 변환 계수 (1픽셀 = 0.1미터)
PIXEL_TO_METER = 0.1

def calculate_direction(current_pos, previous_pos, box):
    if previous_pos is None:
        return 0, None
    
    # 픽셀 단위의 방향 계산
    dx = current_pos[0] - previous_pos[0]
    dy = current_pos[1] - previous_pos[1]
    direction = np.arctan2(dy, dx)
    
    # 바운딩 박스의 각도 계산
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    box_angle = np.arctan2(height, width)
    
    return direction, box_angle

# 상황 감지 함수
def detect_situation(direction, box_angle, previous_direction, previous_status, class_name, position):
    if previous_direction is None:
        return "normal_entry"
    
    # 방향 변화 계산
    direction_change = abs(direction - previous_direction)
    if direction_change > np.pi:
        direction_change = 2 * np.pi - direction_change
    
    # 역주행 감지
    if direction_change > np.pi/2:  # 90도 이상 방향 변화
        return "wrong_way_entry"
    
    # 불법 유턴 감지
    if direction_change > np.pi/4 and direction_change < np.pi/2:  # 45도에서 90도 사이 방향 변화
        return "illegal_u_turn"
    
    # 정상 통과
    return "normal_entry"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO11v 객체 감지
    results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS)
    
    # 객체 카운트 초기화
    object_counts = {class_name: 0 for class_name in CLASSES}
    
    # 현재 프레임의 객체 위치 저장
    current_positions = {}
    
    # ROI 영역 표시
    for roi_idx, roi in enumerate(roi_list):
        # 다각형 그리기
        if len(roi) >= 3:
            pts = np.array(roi, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            # 다각형 내부의 객체 감지
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = CLASSES[class_id] if class_id < len(CLASSES) else None

                    # 유효한 클래스인 경우에만 처리
                    if class_name is None or class_name not in CLASSES:
                        continue

                    # 객체의 중심점이 다각형 내부에 있는지 확인
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    if is_point_in_polygon((center_x, center_y), roi):
                        # 다각형 내의 객체만 표시
                        color = get_class_color(class_name)
                        
                        # 객체의 현재 위치 저장
                        current_positions[(x1, y1, x2, y2)] = (center_x, center_y)
                        
                        # 방향 계산
                        direction = 0
                        box_angle = 0
                        status = "normal_entry"
                        
                        if (x1, y1, x2, y2) in previous_positions:
                            direction, box_angle = calculate_direction(
                                (center_x, center_y), 
                                previous_positions[(x1, y1, x2, y2)],
                                (x1, y1, x2, y2)
                            )
                            
                            # 이전 방향과 상태 가져오기
                            previous_direction = movement_directions.get((x1, y1, x2, y2))
                            previous_status = movement_status.get((x1, y1, x2, y2), "normal_entry")
                            
                            # 상황 감지
                            if class_name.lower() in VEHICLE_CLASSES:
                                status = detect_situation(
                                    direction, 
                                    box_angle, 
                                    previous_direction, 
                                    previous_status,
                                    class_name,
                                    (center_x, center_y)
                                )
                            
                            # 현재 방향과 상태 저장
                            movement_directions[(x1, y1, x2, y2)] = direction
                            movement_status[(x1, y1, x2, y2)] = status
                        
                        # 객체와 상태 표시
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        if class_name.lower() in VEHICLE_CLASSES:
                            cv2.putText(frame, f"Status: {status}", 
                                        (x1, y1 - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # 방향 표시 (화살표)
                        if direction != 0:
                            arrow_length = 30
                            end_x = int(center_x + arrow_length * np.cos(direction))
                            end_y = int(center_y + arrow_length * np.sin(direction))
                            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), color, 2)
                        
                        # 객체 카운트 업데이트
                        object_counts[class_name] += 1

    # 이전 위치 업데이트
    previous_positions = current_positions

    # 현재 그리는 다각형 표시
    if points:
        for i, point in enumerate(points):
            cv2.circle(frame, point, 5, (255, 0, 0), -1)
            if i > 0:
                cv2.line(frame, points[i-1], point, (255, 0, 0), 2)
        if len(points) > 1:
            cv2.line(frame, points[-1], points[0], (255, 0, 0), 2)

    # FPS 표시
    cv2.putText(frame, f"FPS: {TARGET_FPS}", (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 결과 표시
    cv2.imshow('Object Detection', frame)

    # 프레임 간 지연 시간 설정 (초당 100프레임)
    key = cv2.waitKey(int(FRAME_INTERVAL * 1000)) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
