# app.py (Streamlit Web Interface)
import streamlit as st
import cv2
from deepface import DeepFace
import pandas as pd
import time
import requests

# ESP32 Configuration
ESP32_IP = "192.168.1.100"  # Update with your ESP32's IP
ROBOT_CONTROL_URL = f"http://{ESP32_IP}/control"
PAN_TILT_URL = f"http://{ESP32_IP}/pantilt"

# Emotion Tracking
emotion_log = []

def track_emotion(emotion):
    emotion_log.append({
        "timestamp": time.time(),
        "emotion": emotion
    })

def get_emotion_stats():
    df = pd.DataFrame(emotion_log)
    if not df.empty:
        df['duration'] = df['timestamp'].diff().shift(-1).fillna(0)
        return df.groupby('emotion')['duration'].sum().reset_index()
    return pd.DataFrame()

# Robot Control Functions
def send_robot_command(command):
    try:
        requests.get(f"{ROBOT_CONTROL_URL}?cmd={command}", timeout=0.1)
    except:
        pass

def send_pan_tilt(pan, tilt):
    try:
        requests.get(f"{PAN_TILT_URL}?pan={pan}&tilt={tilt}", timeout=0.1)
    except:
        pass

# Streamlit Pages
def emotion_page():
    st.title("Real-time Emotion Detection")
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    
    camera = cv2.VideoCapture(0)
    last_update = time.time()
    
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        
        if time.time() - last_update > 1:  # Process every 1 second
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                track_emotion(emotion)
                st.write(f"Current Emotion: {emotion}")
            except:
                pass
            last_update = time.time()
    else:
        camera.release()

def stats_page():
    st.title("Emotion Statistics")
    stats = get_emotion_stats()
    if not stats.empty:
        st.bar_chart(stats.set_index('emotion'))
        st.write("Detailed Data:")
        st.dataframe(stats)
    else:
        st.write("No data collected yet")

def control_page():
    st.title("Robot Control")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Forward"):
            send_robot_command("forward")
    with col3:
        if st.button("Backward"):
            send_robot_command("backward")
    with col2:
        if st.button("Stop"):
            send_robot_command("stop")
    
    st.subheader("Pan-Tilt Control")
    pan = st.slider("Pan", -90, 90, 0)
    tilt = st.slider("Tilt", -45, 45, 0)
    send_pan_tilt(pan, tilt)

# Main App
pages = {
    "Emotion Detection": emotion_page,
    "Statistics": stats_page,
    "Robot Control": control_page,
}

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(pages.keys()))
pages[page]()