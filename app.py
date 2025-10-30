import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np
from PIL import Image

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="ASL Sign Language Detector", layout="centered")
st.title("🤟 American Sign Language (ASL) → Letter Recognition")
st.write("Upload an **image** or **video** showing a hand sign. The app will detect the ASL letter using your YOLO model!")

# ---------------------------
# Load YOLO Model
# ---------------------------
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file 'best.pt' not found. Please upload it to the same folder as app.py.")
    st.stop()

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

# ---------------------------
# ASL Meaning Dictionary
# ---------------------------
asl_meanings = {
    "A": "The sign for 'A' is made by making a fist with the thumb beside the index finger.",
    "B": "Raise your hand, keep fingers together and thumb across the palm.",
    "C": "Curve your hand to form the letter 'C'.",
    "D": "Index finger points up, thumb and other fingers form a circle.",
    "E": "Curl all fingers toward the palm touching the thumb.",
    "F": "Thumb and index finger make a circle, other fingers raised.",
    "G": "Index finger and thumb extended sideways close together.",
    "H": "Index and middle fingers extended horizontally.",
    "I": "Pinky finger up, rest in a fist.",
    "J": "Make an 'I' and draw a 'J' in the air.",
    "K": "Index and middle fingers raised with thumb between them.",
    "L": "Thumb and index finger make an 'L' shape.",
    "M": "Three fingers over the thumb (like hiding thumb under 3 fingers).",
    "N": "Two fingers over the thumb (thumb under 2 fingers).",
    "O": "Fingers form a circle — the letter 'O'.",
    "P": "Similar to 'K' but tilted downward.",
    "Q": "Like 'G' but pointing down.",
    "R": "Index and middle fingers crossed.",
    "S": "Make a fist with the thumb across the front.",
    "T": "Make a fist, thumb tucked between index and middle fingers.",
    "U": "Two fingers up together (like peace sign closed).",
    "V": "Two fingers apart forming 'V'.",
    "W": "Three fingers up forming 'W'.",
    "X": "Bend the index finger like a hook.",
    "Y": "Thumb and pinky extended outward.",
    "Z": "Draw a 'Z' in the air with your index finger."
}

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name

    # ---------------------------
    # IMAGE Processing
    # ---------------------------
    if file_ext in ["jpg", "jpeg", "png"]:
        st.image(temp_path, caption="📸 Uploaded Image", use_container_width=True)
        st.write("Analyzing image... 🔍")

        results = model.predict(temp_path, verbose=False)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="🧠 Detection Result", use_container_width=True)

        detected = [model.names[int(box.cls)] for box in results[0].boxes]
        if detected:
            letter = detected[0]
            st.success(f"✅ Detected Letter: **{letter}**")
            st.info(asl_meanings.get(letter, "Meaning not found for this letter."))
        else:
            st.warning("⚠️ No sign detected. Try a clearer image or better lighting.")

    # ---------------------------
    # VIDEO Processing
    # ---------------------------
    elif file_ext in ["mp4", "mov", "avi"]:
        st.video(temp_path)
        st.write("Analyzing video... please wait ⏳")

        cap = cv2.VideoCapture(temp_path)
        detected_letters = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, verbose=False)
            detected = [model.names[int(box.cls)] for box in results[0].boxes]
            if detected:
                detected_letters.append(detected[0])

        cap.release()

        if detected_letters:
            final_letter = max(set(detected_letters), key=detected_letters.count)
            st.success(f"✅ Detected Letter: **{final_letter}**")
            st.info(asl_meanings.get(final_letter, "Meaning not found for this letter."))
        else:
            st.warning("⚠️ No signs detected in the video. Try again with a clearer video.")
