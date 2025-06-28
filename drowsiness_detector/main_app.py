import streamlit as st
import cv2
import numpy as np
import torch
from drowsiness_detector.predict_max import FatiguePredictor
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.title("Fatigue Detection")
ALERT_SOUND = "alert.wav"

default_ice_servers = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]} 

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.predictor = FatiguePredictor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.active_array = []
        self.fatigue_array = []
        self.status = "Not Fatigued"
        self.status2 = "Not Fatigued"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        orig = frame.to_ndarray(format="bgr24")
        probs, annotated = self.predictor.predict(orig)

        if isinstance(probs, int) and probs == -1:
            cv2.putText(orig, "No face found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            combined = cv2.hconcat([orig, orig])
            self.status = "No Face"
            self.status2 = "No Face"
            return av.VideoFrame.from_ndarray(combined, format="bgr24")

        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        active_prob, fatigue_prob = probs[0], probs[1]
        self.active_array.append(active_prob)
        self.fatigue_array.append(fatigue_prob)
        if len(self.active_array) > 200:
            self.active_array.pop(0)
            self.fatigue_array.pop(0)

        aprob = sum(self.active_array)
        fprob = sum(self.fatigue_array)
        self.status = f"Not Fatigued {aprob/(aprob + fprob):.2f}" if aprob >= fprob else f"Fatigued {aprob/(aprob + fprob):.2f}"
        self.status2 = "Not Fatigued" if aprob >= fprob else "Fatigued"
        label_text = "Fatigue 😴" if fatigue_prob > active_prob else "Active 😀"
        print(self.status2)
        if self.status2 == "Fatigued":
            st.audio(ALERT_SOUND, format='audio/wav', start_time=0, autoplay=True)

        lines = [label_text,
                 f"Active: {active_prob:.2f}",
                 f"Fatigue: {fatigue_prob:.2f}",
                 f"{self.status}"]
        for i, text in enumerate(lines):
            y = 40 + i * 40
            cv2.putText(orig, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if orig.shape != annotated.shape:
            annotated = cv2.resize(annotated, (orig.shape[1], orig.shape[0]))
        combined = cv2.hconcat([orig, annotated])
        return av.VideoFrame.from_ndarray(combined, format="bgr24")

ctx = webrtc_streamer(
    key="fatigue_detector",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=default_ice_servers
)

if ctx.video_processor:
    status = ctx.video_processor.status2
    print(status)
    if status == "Fatigued":
        st.warning("🛑 Driver appears fatigued! Playing alert sound...")
        # Play sound once when fatigued detected
        st.audio(ALERT_SOUND, format='audio/wav', start_time=0, autoplay=True)
