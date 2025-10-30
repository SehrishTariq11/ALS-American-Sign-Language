import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np

# Streamlit page config
st.set_page_config(page_title="ASL Sign Language Translator", layout="wide")
st.title("🤟 ASL (American Sign Language) → Text Translator")
st.write("Show a hand sign to your webcam — the model will recognize it and display the corresponding alphabet.")

# ---------------------------
# Load YOLO model safely
# ---------------------------
model_path = "best.pt"  # Make sure best.pt is in same folder
if not os.path.exists(model_path):
    st.error(f"Model file not found! Upload your YOLO model as 'best.pt'.")
    st.stop()

model = YOLO(model_path)

# ---------------------------
# Capture video from webcam
# ---------------------------
run = st.checkbox("✅ Start Camera")
FRAME_WINDOW = st.image([])
text_box = st.empty()

detected_text = ""

if run:
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    st.info("Camera is running... show your ASL signs!")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Could not access webcam.")
            break

        # Run YOLO prediction
        results = model.predict(frame, verbose=False)

        # Draw detections on frame
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Get detected classes (signs)
        detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]

        if detected_classes:
            letter = detected_classes[0]
            # Avoid repeating same letter continuously
            if len(detected_text) == 0 or detected_text[-1] != letter:
                detected_text += letter

        # Display
        FRAME_WINDOW.image(annotated_frame, channels="RGB")
        text_box.text_area("📝 Detected Alphabets:", detected_text, height=150)

        # Stop button
        stop = st.button("🛑 Stop Camera")
        if stop:
            break

    cap.release()
    st.success("✅ Camera stopped.")
else:
    st.info("👆 Click the checkbox above to start the camera.")
