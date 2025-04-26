import cv2
import numpy as np
import torch
from roboflow import Roboflow

# Roboflow API 설정
rf = Roboflow(api_key="KLlcHdVtvytxtpDiXA0W")
project = rf.workspace("joyk").project("jyk-hxqi8")
version = project.version(1)
dataset = version.download("yolov11")
model = version.model

# 모델 설정
CONFIDENCE_THRESHOLD = 0.15  # 신뢰도 임계값을 낮춤
IOU_THRESHOLD = 0.35        # IOU 임계값을 낮춤
MAX_DETECTIONS = 100        # 최대 감지 객체 수

# 클래스 정의
CLASSES = [
    # 차량 종류
    'car', 'bus', 'truck', 'motorcycle', 'bicycle',
    # 사람
    'person',
    # 신호등
    'red_light', 'green_light', 'yellow_light',
    # 횡단보도
    'crosswalk'
]

# 클래스별 색상 정의
COLORS = {
    # 차량 관련
    'car': (0, 255, 0),         # 녹색
    'bus': (0, 200, 0),         # 진한 녹색
    'truck': (0, 150, 0),       # 더 진한 녹색
    'motorcycle': (0, 100, 0),  # 가장 진한 녹색
    'bicycle': (0, 50, 0),      # 가장 진한 녹색
    # 사람
    'person': (255, 0, 255),    # 자홍색
    # 신호등
    'red_light': (0, 0, 255),   # 빨간색
    'green_light': (0, 255, 0), # 녹색
    'yellow_light': (0, 255, 255), # 노란색
    # 횡단보도
    'crosswalk': (255, 255, 255) # 흰색
}

# 클래스 카테고리 정의
VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
TRAFFIC_LIGHT_CLASSES = ['red_light', 'green_light', 'yellow_light']

def get_class_color(class_name):
    return COLORS.get(class_name, (255, 255, 255))  # 기본값: 흰색

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

# 객체 카운트를 위한 딕셔너리 초기화
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
def detect_situation(direction, box_angle, previous_direction, previous_status, class_name, position, traffic_light_status=None, pedestrian_positions=None):
    if previous_direction is None:
        return "normal_entry"
    
    # 방향 변화 계산
    direction_change = abs(direction - previous_direction)
    if direction_change > np.pi:
        direction_change = 2 * np.pi - direction_change
    
    # 신호 위반 감지
    if traffic_light_status == 'red_light' and class_name in VEHICLE_CLASSES:
        return "violation_redlight"
    
    # 역주행 감지
    if direction_change > np.pi/2:  # 90도 이상 방향 변화
        return "wrong_way_entry"
    
    # 인도 침입 감지
    if class_name in VEHICLE_CLASSES and is_on_sidewalk(position):
        return "entering_sidewalk"
    
    # 불법 유턴 감지
    if direction_change > np.pi/4 and direction_change < np.pi/2:  # 45도에서 90도 사이 방향 변화
        return "illegal_u_turn"
    
    # 교차로 정지 감지
    if is_in_intersection(position) and is_stopped(position, previous_positions):
        return "blocking_intersection"
    
    # 보행자 위협 감지
    if class_name in VEHICLE_CLASSES and is_near_pedestrian(position, pedestrian_positions):
        return "conflict_pedestrian"
    
    # 정상 통과
    return "normal_entry"

def is_on_sidewalk(position):
    # 인도 영역 체크 로직 (ROI 기반)
    for roi in roi_list:
        if roi.get('type') == 'sidewalk' and is_point_in_polygon(position, roi['points']):
            return True
    return False

def is_in_intersection(position):
    # 교차로 영역 체크 로직 (ROI 기반)
    for roi in roi_list:
        if roi.get('type') == 'intersection' and is_point_in_polygon(position, roi['points']):
            return True
    return False

def is_stopped(position, previous_positions):
    # 정지 상태 체크 로직
    if position in previous_positions:
        prev_pos = previous_positions[position]
        distance = np.sqrt((position[0] - prev_pos[0])**2 + (position[1] - prev_pos[1])**2)
        return distance < 5  # 5픽셀 이내 이동은 정지로 간주
    return False

def is_near_pedestrian(vehicle_position, pedestrian_positions, threshold=50):
    # 보행자 근접 체크 로직
    if pedestrian_positions:
        for ped_pos in pedestrian_positions:
            distance = np.sqrt((vehicle_position[0] - ped_pos[0])**2 + 
                             (vehicle_position[1] - ped_pos[1])**2)
            if distance < threshold:
                return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Roboflow 객체 감지
    try:
        results = model.predict(frame, confidence=CONFIDENCE_THRESHOLD, overlap=IOU_THRESHOLD)
        
        # 객체 카운트 초기화
        object_counts = {class_name: 0 for class_name in CLASSES}
        
        # 현재 프레임의 객체 위치 저장
        current_positions = {}
        
        # 모든 객체에 대해 처리
        for prediction in results:
            x1 = int(prediction['x'] - prediction['width']/2)
            y1 = int(prediction['y'] - prediction['height']/2)
            x2 = int(prediction['x'] + prediction['width']/2)
            y2 = int(prediction['y'] + prediction['height']/2)
            confidence = prediction['confidence']
            class_name = prediction['class']

            # 유효한 클래스인 경우에만 처리
            if class_name not in CLASSES:
                continue

            # 객체의 중심점 계산
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # ROI가 지정된 경우, ROI 내부의 객체만 처리
            if roi_list:
                is_in_roi = False
                for roi in roi_list:
                    if is_point_in_polygon((center_x, center_y), roi):
                        is_in_roi = True
                        break
                if not is_in_roi:
                    continue  # ROI 외부의 객체는 건너뛰기
            
            # 객체 표시
            color = get_class_color(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 객체 정보 표시
            text_y = y1 - 10
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 객체의 현재 위치 저장
            current_positions[(x1, y1, x2, y2)] = (center_x, center_y)
            
            # 방향 계산
            direction = 0
            box_angle = 0
            status = "normal_entry"
            direction_change = 0
            
            if (x1, y1, x2, y2) in previous_positions:
                direction, box_angle = calculate_direction(
                    (center_x, center_y), 
                    previous_positions[(x1, y1, x2, y2)],
                    (x1, y1, x2, y2)
                )
                
                # 이전 방향과 상태 가져오기
                previous_direction = movement_directions.get((x1, y1, x2, y2))
                previous_status = movement_status.get((x1, y1, x2, y2), "normal_entry")
                
                # 방향 변화 계산
                if previous_direction is not None:
                    direction_change = abs(direction - previous_direction)
                    if direction_change > np.pi:
                        direction_change = 2 * np.pi - direction_change
                
                # 상황 감지
                if class_name.lower() in VEHICLE_CLASSES:
                    # 현재 프레임의 신호등 상태와 보행자 위치 수집
                    traffic_light_status = None
                    pedestrian_positions = []
                    for pred in results:
                        if pred['class'] in TRAFFIC_LIGHT_CLASSES:
                            traffic_light_status = pred['class']
                        elif pred['class'] == 'person':
                            ped_center_x = (pred['x'] - pred['width']/2 + pred['x'] + pred['width']/2) // 2
                            ped_center_y = (pred['y'] - pred['height']/2 + pred['y'] + pred['height']/2) // 2
                            pedestrian_positions.append((ped_center_x, ped_center_y))
                    
                    status = detect_situation(
                        direction, 
                        box_angle, 
                        previous_direction, 
                        previous_status,
                        class_name,
                        (center_x, center_y),
                        traffic_light_status,
                        pedestrian_positions
                    )
                    
                    # 차량 정보 추가 표시
                    text_y -= 20
                    cv2.putText(frame, f"Status: {status}", 
                                (x1, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    text_y -= 20
                    cv2.putText(frame, f"Direction: {np.degrees(direction):.1f}°", 
                                (x1, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    text_y -= 20
                    cv2.putText(frame, f"Direction Change: {np.degrees(direction_change):.1f}°", 
                                (x1, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    text_y -= 20
                    cv2.putText(frame, f"Box Angle: {np.degrees(box_angle):.1f}°", 
                                (x1, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 현재 방향과 상태 저장
                movement_directions[(x1, y1, x2, y2)] = direction
                movement_status[(x1, y1, x2, y2)] = status
            
            # 방향 표시 (화살표)
            if direction != 0:
                arrow_length = 30
                end_x = int(center_x + arrow_length * np.cos(direction))
                end_y = int(center_y + arrow_length * np.sin(direction))
                cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), color, 2)
            
            # 객체 카운트 업데이트
            object_counts[class_name] += 1

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        continue

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

    # 저장된 ROI 표시
    for roi in roi_list:
        if len(roi) >= 3:
            pts = np.array(roi, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # 객체 카운트 표시
    count_text_y = 30
    for class_name, count in object_counts.items():
        if count > 0:  # 0개 이상인 클래스만 표시
            color = get_class_color(class_name)
            cv2.putText(frame, f"{class_name}: {count}", 
                        (10, count_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            count_text_y += 30

    # FPS 표시
    cv2.putText(frame, f"FPS: {TARGET_FPS}", (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 결과 표시
    cv2.imshow('Object Detection', frame)

    # 프레임 간 지연 시간 설정
    key = cv2.waitKey(int(FRAME_INTERVAL * 1000)) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
