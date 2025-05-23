import streamlit as st
import av
import cv2
import numpy as np
import tempfile
from datetime import datetime
import pyttsx3
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from core import load_saved_model, predict
import pythoncom

model = load_saved_model(load_last=True)
last_alert = datetime.min

def get_tts_engine():
    """
    Initialize COM and create a pyttsx3 engine on demand.
    """
    pythoncom.CoInitialize()
    eng = pyttsx3.init()
    eng.setProperty('rate', 150)
    return eng  

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

st.set_page_config(page_title="Driver Drowsiness", page_icon="🚗💤", layout="wide")

with st.sidebar:
    st.image("./Assets/Logo.png", use_column_width=True)
    st.title("Driver Drowsiness")
    st.markdown("Prevent accidents by alerting the driver in real time or from video uploads.")
    choice = st.radio("Menu", ["Upload Video", "Real-Time Detection"], label_visibility='collapsed')

if choice == "Upload Video":
    st.header("📤 Upload Video Prediction")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        video_file = st.file_uploader("Choose an MP4 video for analysis", type="mp4")
        if video_file:
            with st.spinner("Processing video, please wait..."):
                result = predict_video(video_file)
            st.success("Processing complete!")
            st.video(result)

else:
    st.header("🎥 Real-Time Drowsiness Detection")
    cols = st.columns([1, 6, 1])
    with cols[1]:
        st.markdown("---")
        st.write("Streaming from your webcam. Close the stream to return.")
        def callback(frame):
            global last_alert
            img = frame.to_ndarray(format="bgr24")
            if predict_rt(img):
                alert_img = cv2.addWeighted(img, 0.7, np.full_like(img, (0,0,255)), 0.3, 0)
                now = datetime.now()
                if (now - last_alert).total_seconds() > 5:
                    engine = get_tts_engine()
                    engine.say("WAKE UP! WAKE UP!")
                    engine.runAndWait()
                    last_alert = now
                return av.VideoFrame.from_ndarray(alert_img, format="bgr24")
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        webrtc_streamer(
            key="drowsiness",
            video_frame_callback=callback,
            video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=True)
        )
