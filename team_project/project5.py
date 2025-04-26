import cv2
import numpy as np
import os
import winsound
from collections import deque
from ultralytics import YOLO  # 최신 YOLOv8 API

# 모델 로드
model = YOLO('yolo11n.pt')
CLASSES = model.names

NO_VECTOR_CLASSES = ['person', 'red_light', 'green_light', 'yellow_light', 'crosswalk']

COLORS = {
    'car': (0, 255, 0), 'bus': (0, 200, 0), 'truck': (0, 150, 0), 'motorcycle': (0, 100, 0),
    'person': (255, 0, 255),
    'red_light': (0, 0, 255), 'green_light': (0, 255, 0), 'yellow_light': (0, 255, 255),
    'crosswalk': (255, 255, 255),
    'violation_redlight': (0, 0, 255), 'wrong_way_entry': (0, 100, 255),
    'entering_sidewalk': (255, 100, 0), 'illegal_u_turn': (255, 0, 100),
    'blocking_intersection': (200, 0, 100), 'conflict_pedestrian': (150, 0, 255),
    'normal_entry': (0, 255, 0)
}

def get_class_color(class_name):
    return COLORS.get(class_name, (255, 255, 255))

class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.tracks = {}

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

            unused_cols = set(range(len(input_centroids))).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])

            unused_rows = set(range(len(obj_centroids))).difference(used_rows)
            for row in unused_rows:
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

# 비디오 로드
video_path = "video/KakaoTalk_20250415_123136238.mp4"
if not os.path.exists(video_path):
    print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
cv2.namedWindow('Violation Detection with Alert')

# 저장할 폴더 및 파일명 설정
output_dir = 'result_video'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'YOLO11n.avi')

# 저장할 비디오 설정
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

click_points = []
base_vectors = []
selecting_base = True

def click_event(event, x, y, flags, param):
    global click_points, base_vectors, selecting_base
    if selecting_base and event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        if len(click_points) % 2 == 0:
            pt1, pt2 = click_points[-2], click_points[-1]
            base_vectors.append(np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]]))
    elif event == cv2.EVENT_RBUTTONDOWN:
        selecting_base = False

cv2.setMouseCallback('Violation Detection with Alert', click_event)

trackers = {cls: CentroidTracker() for cls in CLASSES if cls not in NO_VECTOR_CLASSES or cls == 'person'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    alarm_triggered = False

    try:
        results = model.predict(source=frame, conf=0.25, verbose=False)[0]
        detections = {cls: [] for cls in trackers}

        for result in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = result
            class_name = CLASSES[int(cls_id)]
            if conf < 0.25:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            color = get_class_color(class_name)
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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

                if cls != 'person':
                    cv2.circle(frame, center, 4, color, -1)
                    if direction:
                        dx = int(30 * np.cos(direction))
                        dy = int(30 * np.sin(direction))
                        cv2.arrowedLine(frame, center, (center[0] + dx, center[1] + dy), color, 2)

                label = f"{cls} ID:{object_id}"
                cv2.putText(frame, label, (center[0] - 10, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for p in obj_positions.get('person', {}).values():
            for vehicle_cls in ['car', 'bus', 'truck', 'motorcycle']:
                for v in obj_positions.get(vehicle_cls, {}).values():
                    if compute_distance(p, v) < 80:
                        cv2.line(frame, p, v, (0, 0, 255), 2)
                        cv2.putText(frame, "conflict_pedestrian", (p[0], p[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        alarm_triggered = True

    except Exception as e:
        print(f"오류: {e}")

    if alarm_triggered:
        winsound.Beep(1000, 200)

    resized_frame = cv2.resize(frame, (640, 320))
    cv2.imshow('Violation Detection with Alert', resized_frame)

    # 저장용 원본 프레임 기록
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
