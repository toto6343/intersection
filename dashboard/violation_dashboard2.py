import streamlit as st
import cv2
import tempfile
import os
from roboflow import Roboflow
import numpy as np
from twilio.rest import Client
import plotly.express as px
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# .env 환경변수 불러오기
load_dotenv()
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_FROM = os.getenv("TWILIO_FROM")
TWILIO_TO = os.getenv("TWILIO_TO")

# Streamlit 설정
st.set_page_config(layout="wide")
st.title("🚦 실시간 위반 탐지 대시보드")

# Roboflow 모델 로딩 및 YOLOv11 데이터셋 다운로드
rf = Roboflow(api_key="EOcgTkCLUc6sFR8Pv6Lf")
project = rf.workspace("joyk-cl8nt").project("project-twhf4")
version = project.version(1)
model = version.model
dataset = version.download("yolov11")  # 로컬로 데이터셋 다운로드

# 사용자 영상 업로드
video_file = st.file_uploader("🔼 영상 업로드 (.mp4, .avi)", type=["mp4", "avi"])

# 최대 프레임 수 설정
max_frames = st.slider("🎞️ 처리할 최대 프레임 수", 100, 2000, 300)

# 탐지 임계값
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# 문자 알림 함수
def send_sms_alert(message):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        client.messages.create(
            body=message,
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        st.success("📱 Twilio 문자 전송 완료!")
    except Exception as e:
        st.error(f"❌ 문자 전송 실패: {e}")

# 로그 저장용 리스트
log_records = []
true_labels = []
predicted_labels = []
confidences = []

# 영상 처리 함수
def process_video(video_path, max_frames):
    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()
    stats = {}
    frame_count = 0
    alert_sent = False
    os.makedirs("captures", exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > max_frames:
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

                stats[cls] = stats.get(cls, 0) + 1

                if cls in ["car", "bus", "truck"] and not alert_sent:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"captures/violation_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    send_sms_alert(f"🚨 위반 차량 감지됨 - 이미지 저장 완료: {filename}")
                    alert_sent = True

                log_records.append({
                    "frame": frame_count,
                    "class": cls,
                    "confidence": round(conf, 2),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                true_labels.append(cls)
                predicted_labels.append(cls)
                confidences.append(conf)

                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{cls} {conf:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        except Exception as e:
            st.warning(f"예측 중 오류 발생: {e}")

        frame = cv2.resize(frame, (720, 480))
        frame_placeholder.image(frame, channels="BGR")
        frame_count += 1

    cap.release()
    return stats

# 메인 실행
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    stats = process_video(tfile.name, max_frames)

    # 탐지 통계
    st.subheader("📊 탐지 통계")
    st.write(stats)

    # 로그 저장
    df_logs = pd.DataFrame(log_records)
    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_logs.to_csv(csv_path, index=False)
    st.success(f"📁 CSV 로그 저장 완료: `{csv_path}`")

    # 평가 지표 계산
    if true_labels and predicted_labels:
        precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        cm = confusion_matrix(true_labels, predicted_labels)

        # 이진화 (위반 차량 vs 기타)
        true_labels_binary = [1 if label in ["car", "bus", "truck"] else 0 for label in true_labels]
        confidences_array = np.array(confidences)

        st.subheader("📊 모델 평가 지표")
        st.write(f"정확도 (Accuracy): {accuracy:.2f}")
        st.write(f"정밀도 (Precision): {precision:.2f}")
        st.write(f"재현율 (Recall): {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")

        if len(confidences_array) > 0 and len(set(true_labels_binary)) > 1:
            average_precision = average_precision_score(true_labels_binary, confidences_array)
            precision_vals, recall_vals, _ = precision_recall_curve(true_labels_binary, confidences_array)
            fpr, tpr, _ = roc_curve(true_labels_binary, confidences_array)
            st.write(f"평균 정밀도 (Average Precision): {average_precision:.2f}")

            st.subheader("📊 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)

            st.subheader("📈 Precision-Recall 곡선")
            fig, ax = plt.subplots()
            ax.plot(recall_vals, precision_vals, color='blue')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            st.pyplot(fig)

            st.subheader("📈 ROC 곡선")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            st.pyplot(fig)
        else:
            st.warning("📉 PR/ROC 곡선을 생성할 수 없습니다 (데이터 부족)")

        st.subheader("📈 신뢰 점수 분포")
        fig = px.histogram(df_logs, x="confidence", title="신뢰 점수 분포")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("📉 평가를 위한 라벨 데이터가 충분하지 않습니다.")

    st.success("✅ 영상 처리 완료!")
