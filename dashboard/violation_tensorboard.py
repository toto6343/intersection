import streamlit as st
import cv2
import tempfile
from roboflow import Roboflow
import numpy as np
from torch.utils.tensorboard import SummaryWriter

st.set_page_config(layout="wide")

# 제목
st.title("🚦 실시간 위반 탐지 대시보드")

# 영상 업로드
video_file = st.file_uploader("🔼 영상 업로드 (.mp4)", type=["mp4", "avi"])

# Roboflow 모델 불러오기
rf = Roboflow(api_key="EOcgTkCLUc6sFR8Pv6Lf")
project = rf.workspace("joyk-cl8nt").project("project-twhf4")
version = project.version(1)
model = version.model

# 탐지 임계값 설정
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# TensorBoard 로그 기록 객체
writer = SummaryWriter(log_dir="runs/object_detection_log")

# 영상 처리 함수
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()
    stats = {"person": 0, "car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > 300:  # 프레임 수 제한
            break

        try:
            results = model.predict(frame, confidence=CONFIDENCE_THRESHOLD, overlap=IOU_THRESHOLD)
            predictions = results.json()["predictions"]

            for pred in predictions:
                cls = pred["class"].lower()
                x = int(pred['x'] - pred['width'] / 2)
                y = int(pred['y'] - pred['height'] / 2)
                w = int(pred['width'])
                h = int(pred['height'])
                conf = pred['confidence']

                if cls in stats:
                    stats[cls] += 1

                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{cls} {conf:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        except Exception as e:
            st.warning(f"예측 중 오류 발생: {e}")

        # 프레임 사이즈 조정 및 출력
        frame = cv2.resize(frame, (720, 480))
        frame_placeholder.image(frame, channels="BGR")

        # TensorBoard 로그 기록
        writer.add_scalar("Detection/Person", stats["person"], frame_count)
        writer.add_scalar("Detection/Car", stats["car"], frame_count)
        writer.add_scalar("Detection/Bus", stats["bus"], frame_count)
        writer.add_scalar("Detection/Truck", stats["truck"], frame_count)
        writer.add_scalar("Detection/Motorcycle", stats["motorcycle"], frame_count)

        frame_count += 1

    cap.release()
    writer.close()  # 로그 기록 종료
    return stats

# 업로드된 비디오 처리
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    stats = process_video(tfile.name)

    # 결과 통계 표시
    st.subheader("📊 탐지 통계")
    st.write(stats)

    st.success("✅ 영상 처리 완료! TensorBoard에서 확인해보세요.")

    st.info("💡 터미널에 아래 명령어 입력 후 웹 브라우저에서 TensorBoard를 열어보세요:\n\n```bash\ntensorboard --logdir=runs\n```")
