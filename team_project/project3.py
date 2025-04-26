import cv2
import numpy as np
import os
from roboflow import Roboflow
from playsound import playsound
import threading

# Roboflow API 설정
rf = Roboflow(api_key="KLlcHdVtvytxtpDiXA0W")
project = rf.workspace("joyk").project("jyk-jipji")
version = project.version(1)
dataset = version.download("yolov11")

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

CLASSES = [
    'car', 'bus', 'truck', 'motorcycle',
    'person',
    'red_light', 'green_light', 'yellow_light',
    'crosswalk',
    'violation_redlight', 'wrong_way_entry', 'entering_sidewalk',
    'illegal_u_turn', 'blocking_intersection', 'conflict_pedestrian',
    'normal_entry'
]

COLORS = {
    'car': (0, 255, 0),
    'bus': (0, 200, 0),
    'truck': (0, 150, 0),
    'motorcycle': (0, 100, 0),
    'person': (255, 0, 255),
    'red_light': (0, 0, 255),
    'green_light': (0, 255, 0),
    'yellow_light': (0, 255, 255),
    'crosswalk': (255, 255, 255),
    'violation_redlight': (255, 0, 0),
    'wrong_way_entry': (255, 0, 0),
    'entering_sidewalk': (255, 0, 0),
    'illegal_u_turn': (255, 0, 0),
    'blocking_intersection': (255, 0, 0),
    'conflict_pedestrian': (255, 0, 0),
    'normal_entry': (0, 255, 0)
}

WARNING_BEHAVIORS = {
    'violation_redlight', 'wrong_way_entry', 'entering_sidewalk',
    'illegal_u_turn', 'blocking_intersection', 'conflict_pedestrian'
}

def get_class_color(class_name):
    return COLORS.get(class_name, (255, 255, 255))

video_path = "video\KakaoTalk_20250415_123136238.mp4"
if not os.path.exists(video_path):
    print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("비디오 캡처를 시작할 수 없습니다.")
    exit(1)

cv2.namedWindow('Object Detection')

click_points = []
base_vectors = []
selecting_base = True

def click_event(event, x, y, flags, param):
    global click_points, base_vectors, selecting_base
    if selecting_base and event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f"기준점 선택됨: {len(click_points)}개")
        if len(click_points) % 2 == 0:
            pt1, pt2 = click_points[-2], click_points[-1]
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            vec = np.array([dx, dy])
            base_vectors.append(vec)
            print(f"기준 벡터 추가됨: {vec}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        selecting_base = False
        print("기준 벡터 설정 완료!")

cv2.setMouseCallback('Object Detection', click_event)

object_counts = {class_name: 0 for class_name in CLASSES}
previous_positions = {}
frame_count = 0

def calculate_direction(prev_pos, curr_pos):
    if prev_pos is None:
        return None
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    angle = np.arctan2(dy, dx)
    return angle

def analyze_behavior(class_name, direction, position, frame_shape):
    behaviors = []
    if not base_vectors or direction is None:
        return behaviors

    if class_name in ['car', 'bus', 'truck', 'motorcycle']:
        move_vector = np.array([np.cos(direction), np.sin(direction)])
        is_normal = False

        for base_vec in base_vectors:
            base_unit = base_vec / np.linalg.norm(base_vec)
            angle_between = np.degrees(np.arccos(np.clip(np.dot(move_vector, base_unit), -1.0, 1.0)))
            if angle_between <= 90:
                is_normal = True
            else:
                behaviors.append('wrong_way_entry')
                break

        if is_normal and 'wrong_way_entry' not in behaviors:
            behaviors.append('normal_entry')

        if position[1] < frame_shape[0] * 0.2:
            behaviors.append('entering_sidewalk')

    elif class_name == 'person':
        if any(obj_class in ['car', 'bus', 'truck', 'motorcycle']
               for obj_class in previous_positions.keys()):
            behaviors.append('conflict_pedestrian')

    return behaviors

def play_warning_sound():
    threading.Thread(target=playsound, args=('alert.wav',), daemon=True).start()

print("프로그램을 시작합니다...")
print("🚗 정주행 기준 방향을 설정하세요:")
print(" - 왼쪽 클릭: 점 추가 (2점씩 1쌍)")
print(" - 오른쪽 클릭: 설정 종료")
print("'q' 키를 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("비디오 종료")
        break

    frame_count += 1
    object_counts = {class_name: 0 for class_name in CLASSES}
    behavior_counts = {behavior: 0 for behavior in WARNING_BEHAVIORS.union({'normal_entry'})}

    warning_triggered = False

    try:
        results = version.model.predict(frame, confidence=CONFIDENCE_THRESHOLD, overlap=IOU_THRESHOLD)
        if results:
            predictions = results.json()['predictions']

            for pred in predictions:
                x = int(pred['x'] - pred['width'] / 2)
                y = int(pred['y'] - pred['height'] / 2)
                w = int(pred['width'])
                h = int(pred['height'])
                confidence = pred['confidence']
                class_name = pred['class'].lower()

                if class_name in ['bus', 'truck'] and confidence < 0.9:
                    continue
                if class_name in ['person', 'motorcycle', 'bicycle'] and confidence < 0.5:
                    continue

                if class_name in CLASSES:
                    current_pos = (x + w / 2, y + h / 2)
                    prev_pos = previous_positions.get(class_name)
                    direction = calculate_direction(prev_pos, current_pos)
                    behaviors = analyze_behavior(class_name, direction, current_pos, frame.shape)

                    for behavior in behaviors:
                        if behavior in behavior_counts:
                            behavior_counts[behavior] += 1
                            if behavior in WARNING_BEHAVIORS:
                                warning_triggered = True

                    color = get_class_color(class_name)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    if direction is not None:
                        arrow_length = 30
                        end_x = int(current_pos[0] + arrow_length * np.cos(direction))
                        end_y = int(current_pos[1] + arrow_length * np.sin(direction))
                        cv2.arrowedLine(frame, (int(current_pos[0]), int(current_pos[1])),
                                        (end_x, end_y), color, 2)
                        angle_deg = np.degrees(direction)
                        cv2.putText(frame, f"{angle_deg:.1f}°", (int(current_pos[0]), int(current_pos[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    label = f"{class_name}: {confidence:.2f}"
                    if behaviors:
                        label += f" ({', '.join(behaviors)})"
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x, y - label_height - 10), (x + label_width, y), color, -1)
                    cv2.putText(frame, label, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    object_counts[class_name] += 1
                    previous_positions[class_name] = current_pos

    except Exception as e:
        print(f"객체 감지 중 오류 발생: {str(e)}")
        continue

    # 경고음 재생
    if warning_triggered:
        play_warning_sound()

    info_y = 30
    line_height = 30

    cv2.putText(frame, f"Frame: {frame_count}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_y += line_height
    total_objects = sum(object_counts.values())
    cv2.putText(frame, f"Objects: {total_objects}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_y += line_height
    total_behaviors = sum(behavior_counts.values())
    cv2.putText(frame, f"Behaviors: {total_behaviors}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_y += line_height

    for behavior, count in behavior_counts.items():
        if count > 0:
            color = (0, 255, 0) if behavior == 'normal_entry' else (0, 0, 255)
            cv2.putText(frame, f"{behavior}: {count}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            info_y += line_height

    for i in range(0, len(click_points) - 1, 2):
        cv2.arrowedLine(frame, click_points[i], click_points[i+1], (255, 255, 0), 3)

    cv2.imshow('Object Detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("프로그램을 종료합니다.")
