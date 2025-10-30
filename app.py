import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

st.set_page_config(page_title="ASL to Text Translator", layout="wide")

st.title("🤟 American Sign Language (ASL) to Text Translator")
st.write("This app uses YOLOv8 to detect ASL signs in real time from your webcam and convert them into text.")

# Load YOLO model
model = YOLO("best.pt")  # replace with your trained model file

# Initialize webcam
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
text_placeholder = st.empty()

detected_text = ""  # stores recognized letters

st.write("▶️ The webcam is now running... make sure your camera permission is allowed.")

stop_btn = st.button("🛑 Stop Stream")

while cap.isOpened() and not stop_btn:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to access the webcam.")
        break

    # Predict with YOLO
    results = model.predict(frame, verbose=False)

    # Annotate video frame
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

    # Get class name(s) predicted
    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]

    # Add detected letters to text box (if new)
    if detected_classes:
        letter = detected_classes[0]
        if len(detected_text) == 0 or detected_text[-1] != letter:
            detected_text += letter

    # Show text box
    text_placeholder.text_area("📝 Detected Text:", detected_text, height=150)

cap.release()
st.success("✅ Stream stopped successfully!")

