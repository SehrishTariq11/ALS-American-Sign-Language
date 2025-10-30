import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

st.title("Real-Time ASL Detection")
st.write("Webcam captures your gestures and predicts the corresponding alphabet.")

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # replace with your model path

# Webcam capture
FRAME_WINDOW = st.image([])

# OpenCV video capture
cap = cv2.VideoCapture(0)  # 0 = default webcam

st.write("Press 'Stop' to end the session.")
stop_button = st.button("Stop")

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    # YOLO prediction
    results = model.predict(frame, verbose=False)

    # Annotate frame
    annotated_frame = results[0].plot()
    
    # Convert BGR to RGB for Streamlit
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    # Show video frame
    FRAME_WINDOW.image(annotated_frame)
    
    # Extract detected class names (alphabet letters)
    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    if detected_classes:
        st.text(f"Detected Sign(s): {', '.join(detected_classes)}")
