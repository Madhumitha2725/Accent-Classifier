import streamlit as st
import librosa
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load model (make sure model.pkl is in the same folder or use your own path)
with open("accent_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature extraction function
def extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        return None

# Streamlit UI
st.set_page_config(page_title="Accent Detection App", layout="centered")
st.title("🗣️ English Accent Detection App")
st.write("Upload a voice clip (WAV format) and I’ll predict the English accent!")

uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    features = extract_mfcc("temp.wav")
    if features is not None:
        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])
        confidence = np.max(proba) * 100
        st.success(f"Predicted Accent: **{prediction.capitalize()}** 🎯")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.error("Could not process the audio. Please try again with a clear .wav file.")
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import soundfile as sf

st.subheader("🎤 Or record your voice (3 seconds recommended)")

class AudioProcessor:
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame):
        self.audio_frames.append(frame.to_ndarray())
        return frame

ctx = webrtc_streamer(
    key="sendonly-audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

if ctx.audio_receiver:
    audio_processor = AudioProcessor()
    audio_frames = ctx.audio_receiver.get_frames(timeout=2)

    if audio_frames:
        for frame in audio_frames:
            audio_processor.recv(frame)

        # Combine and save the audio
        audio_data = np.concatenate(audio_processor.audio_frames, axis=0)
        temp_path = "recorded.wav"
        sf.write(temp_path, audio_data, 48000)

        st.audio(temp_path)

        features = extract_mfcc(temp_path)
        if features is not None:
            prediction = model.predict([features])[0]
            confidence = np.max(model.predict_proba([features])) * 100
            st.success(f"Predicted Accent: **{prediction.capitalize()}** 🎯")
            st.info(f"Confidence: {confidence:.2f}%")
