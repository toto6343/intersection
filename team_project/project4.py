import cv2
import numpy as np
import os
import winsound
from roboflow import Roboflow
from collections import deque

# Roboflow API 연결 및 데이터셋 다운로드 (선택적)
rf = Roboflow(api_key="EOcgTkCLUc6sFR8Pv6Lf")
project = rf.workspace("joyk-cl8nt").project("project-twhf4")
version = project.version(1)
model = version.model
dataset = version.download("yolov11")  # <-- 이 부분은 학습 데이터를 다운로드할 때만 사용

# 설정값
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DISAPPEARED = 30

# 색상 정의
COLORS = {
    'car': (0, 255, 0), 'bus': (0, 200, 0), 'truck': (0, 150, 0), 'motorcycle': (0, 100, 0),
    'person': (255, 0, 255), 'red_light': (0, 0, 255), 'green_light': (0, 255, 0), 'yellow_light': (0, 255, 255),
    'crosswalk': (255, 255, 255), 'violation_redlight': (0, 0, 255), 'wrong_way_entry': (0, 100, 255),
    'entering_sidewalk': (255, 100, 0), 'illegal_u_turn': (255, 0, 100), 'blocking_intersection': (200, 0, 100),
    'conflict_pedestrian': (150, 0, 255), 'normal_entry': (0, 255, 0)
}

def get_class_color(class_name):
    return COLORS.get(class_name, (255, 255, 255))

class CentroidTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED):
        self.next_object_id = 0
        self.objects = dict()
        self.disappeared = dict()
        self.tracks = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.tracks[self.next_object_id] = deque(maxlen=10)
        self.tracks[self.next_object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.tracks[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects, self.tracks

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(obj_centroids)[:, None] - np.array(input_centroids), axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                obj_id = obj_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.tracks[obj_id].append(input_centroids[col])
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            for col in set(range(len(input_centroids))).difference(used_cols):
                self.register(input_centroids[col])
            for row in set(range(len(obj_centroids))).difference(used_rows):
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

        return self.objects, self.tracks

def compute_direction(track):
    if len(track) < 2:
        return None
    dx = track[-1][0] - track[0][0]
    dy = track[-1][1] - track[0][1]
    return np.arctan2(dy, dx)

def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 입력 비디오
video_path = "video/KakaoTalk_20250415_123136238.mp4"
if not os.path.exists(video_path):
    print("❌ 비디오 파일 없음")
    exit(1)

cap = cv2.VideoCapture(video_path)
fps = 30  # 강제 설정
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 출력 비디오 설정
output_dir = "result_video"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "result.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 트래커 초기화
cv2.namedWindow('Violation Detection with Alert')
trackers = {cls: CentroidTracker() for cls in COLORS.keys()}

frame_count = 0

try:
    while frame_count < 1000:
        ret, frame = cap.read()
        if not ret:
            print("✅ 영상 끝")
            break

        alarm_triggered = False

        try:
            results = model.predict(frame, confidence=CONFIDENCE_THRESHOLD, overlap=IOU_THRESHOLD)
            predictions = results.json()['predictions']
            detections = {cls: [] for cls in trackers}

            for pred in predictions:
                class_name = pred['class'].lower()
                confidence = pred['confidence']
                if class_name in ['bus', 'truck'] and confidence < 0.9:
                    continue
                if confidence < 0.2:
                    continue

                x = int(pred['x'] - pred['width'] / 2)
                y = int(pred['y'] - pred['height'] / 2)
                w = int(pred['width'])
                h = int(pred['height'])
                center = (x + w // 2, y + h // 2)

                color = get_class_color(class_name)
                label = f"{class_name} {confidence:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if class_name in trackers:
                    detections[class_name].append(center)

            obj_positions = {}
            for cls, tracker in trackers.items():
                objects, tracks = tracker.update(detections[cls])
                obj_positions[cls] = objects

                for object_id, center in objects.items():
                    track = tracks[object_id]
                    direction = compute_direction(track)
                    color = get_class_color(cls)

                    if cls not in ['person', 'crosswalk']:
                        cv2.circle(frame, center, 4, color, -1)
                        if direction:
                            dx = int(30 * np.cos(direction))
                            dy = int(30 * np.sin(direction))
                            cv2.arrowedLine(frame, center, (center[0] + dx, center[1] + dy), color, 2)

                    label = f"{cls} ID:{object_id}"
                    cv2.putText(frame, label, (center[0] - 10, center[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            for p in obj_positions.get('person', {}).values():
                for vehicle_cls in ['car', 'bus', 'truck', 'motorcycle']:
                    for v in obj_positions.get(vehicle_cls, {}).values():
                        if compute_distance(p, v) < 80:
                            cv2.line(frame, p, v, (0, 0, 255), 2)
                            cv2.putText(frame, "conflict_pedestrian", (p[0], p[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            alarm_triggered = True

        except Exception as e:
            print("예측 오류:", e)

        if alarm_triggered:
            winsound.Beep(1000, 200)

        out.write(frame)
        display_frame = cv2.resize(frame, (640, 480))
        cv2.imshow('Violation Detection with Alert', display_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 사용자 종료")
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"🎥 저장된 프레임 수: {frame_count}")
    print(f"📁 저장 위치: {output_path}")
