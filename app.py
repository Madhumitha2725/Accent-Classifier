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
from streamlit_audio_recorder import audio_recorder
import tempfile

st.subheader("🎤 Or record your voice")

audio_bytes = audio_recorder()

if audio_bytes:
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        temp_audio_path = f.name

    st.audio(temp_audio_path, format='audio/wav')

    features = extract_mfcc(temp_audio_path)
    if features is not None:
        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])
        confidence = np.max(proba) * 100
        st.success(f"Predicted Accent: **{prediction.capitalize()}** 🎯")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.error("Could not process the recorded audio.")
