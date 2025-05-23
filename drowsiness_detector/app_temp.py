# drowsiness_detector/app.py
import streamlit as st
import av
import cv2
import numpy as np
import tempfile
import random
import os
from datetime import datetime
import pyttsx3
import pythoncom
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from core import load_saved_model, predict

# Load model
model = load_saved_model(load_last=True)
# Global state for alert timing and drowsy frame counting
last_alert_time = datetime.min
red_overlay = None
DROWSY_THRESHOLD = 10  # number of consecutive drowsy frames
consecutive_drowsy = 0

# Text-to-Speech setup
def get_tts_engine():
    try:
        pythoncom.CoInitialize()
    except Exception:
        pass
    eng = pyttsx3.init()
    eng.setProperty('rate', 150)
    return eng

# Alarm sound loader
alarm_bytes = None
alarm_path = os.path.join('Assets', 'alarm.wav')
if os.path.exists(alarm_path):
    with open(alarm_path, 'rb') as f:
        alarm_bytes = f.read()

# Trivia bank for sidebar
TRIVIA = [
    "Did you know? Yawning helps cool your brain!",
    "Tip: Rolling down the window increases oxygen flow.",
    "Trivia: The world’s longest recorded yawn lasted 6 minutes!",
    "Fact: Micro-naps of 10-20 seconds can boost alertness."
]

# Prediction functions

def predict_rt(frame):
    return predict(model, frame)

def predict_video(video_file):
    temp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_vid.write(video_file.read())
    temp_vid.close()
    cap = cv2.VideoCapture(temp_vid.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    overlay = np.full((h, w, 3), (0, 0, 255), dtype=np.uint8)
    out_path = f"{video_file.name}_pred.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if predict_rt(frame):
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        out.write(frame)
    cap.release(); out.release()
    return out_path

# Streamlit UI setup
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗💤",
    layout="wide"
)

with st.sidebar:
    st.image("./Assets/Logo.png", use_container_width=True)
    st.title("Driver Drowsiness Detection")
    st.markdown("Prevent accidents by alerting any sustained drowsiness.")
    mode = st.radio("Mode", ["Upload Video", "Real-Time Detection"], label_visibility='collapsed')

# Upload Video Mode
if mode == "Upload Video":
    st.header("📤 Upload Video Prediction")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        video_file = st.file_uploader("Choose an MP4 video", type="mp4")
        if video_file:
            with st.spinner("Processing video..."):
                out = predict_video(video_file)
            st.success("Detection complete!")
            st.video(out)

# Real-Time Mode
else:
    st.header("🎥 Real-Time Drowsiness Detection")
    st.write("Streaming from your webcam. Stay alert!")

    def callback(frame):
        global red_overlay, last_alert_time, consecutive_drowsy
        img = frame.to_ndarray(format="bgr24")
        now = datetime.now()

        # Initialize overlay once
        if red_overlay is None:
            red_overlay = np.full_like(img, (0, 0, 255))

        # Check prediction
        if predict_rt(img):
            consecutive_drowsy += 1
        else:
            consecutive_drowsy = 0

        # Trigger alert if sustained drowsiness
        if consecutive_drowsy >= DROWSY_THRESHOLD and (now - last_alert_time).total_seconds() > 5:
            # Sound alert
            if alarm_bytes:
                st.audio(alarm_bytes, format='audio/wav')
            else:
                engine = get_tts_engine()
                engine.say("Wake up! Wake up!")
                engine.runAndWait()

            # Visual alert
            st.balloons()
            st.sidebar.info(f"💡 {random.choice(TRIVIA)}")
            st.sidebar.warning("🔴 Sustained drowsiness detected! Press below if you're awake.")
            if st.sidebar.button("⭕️ I'm Awake", key=f"awake_{now.timestamp()}"):
                st.sidebar.success("👍 Good job! Stay focused.")
                last_alert_time = now
                consecutive_drowsy = 0

            # Flash red overlay
            img = cv2.addWeighted(img, 0.7, red_overlay, 0.3, 0)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="drowsiness",
        video_frame_callback=callback,
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=True),
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False}
    )
