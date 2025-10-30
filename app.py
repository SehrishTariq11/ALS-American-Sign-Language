import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np
from PIL import Image

# --------------------------
# Streamlit Page Settings
# --------------------------
st.set_page_config(page_title="ASL Sign Language Translator", layout="centered")
st.title("🤟 ASL (American Sign Language) → Text Translator")
st.write("Upload an **image or short video** of your ASL sign to recognize the corresponding alphabet.")

# --------------------------
# Load YOLO Model
# --------------------------
model_path = "best.pt"
if not os.path.exists(model_path):
    st.error("❌ 'best.pt' model file not found! Upload it to your app folder.")
    st.stop()

model = YOLO(model_path)

# --------------------------
# Helper: ASL Meanings
# --------------------------
asl_meanings = {
    "A": "The sign for 'A' is made by making a fist with the thumb beside the index finger.",
    "B": "The sign for 'B' is made by raising the palm with fingers together and thumb across the palm.",
    "C": "The hand forms a 'C' shape, as if holding a cup.",
    "D": "The index finger points up while other fingers form a circle.",
    "E": "The fingers curl down to touch the thumb, forming an 'E'.",
    "F": "The index finger and thumb make a circle; other fingers are raised.",
    "G": "The thumb and index finger point sideways close together.",
    "H": "Two fingers extended horizontally, like showing ‘2’.",
    "I": "The pinky is raised while the rest of the hand is in a fist.",
    "J": "Make 'I' sign, then draw a 'J' in the air.",
    "K": "Index and middle finger raised, thumb between them.",
    "L": "Thumb and index finger make an 'L' shape.",
    "M": "Three fingers over the thumb (thumb under 3 fingers).",
    "N": "Two fingers over the thumb (thumb under 2 fingers).",
    "O": "Fingers form an ‘O’ shape.",
    "P": "Similar to 'K' but tilted downward.",
    "Q": "Similar to 'G' but tilted downward.",
    "R": "Index and middle fingers crossed.",
    "S": "Make a fist with thumb across the front.",
    "T": "Make a fist, thumb tucked between index and middle finger.",
    "U": "Two fingers together pointing upward.",
    "V": "Two fingers apart forming a ‘V’.",
    "W": "Three fingers up forming a 'W'.",
    "X": "Index finger bent like a hook.",
    "Y": "Thumb and pinky extended outward.",
    "Z": "Draw the letter 'Z' in the air with your index finger."
}

# --------------------------
# File Upload Section
# --------------------------
uploaded_file = st.file_uploader("📂 Upload an image or video file", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name

    # --------------------------
    # If image uploaded
    # --------------------------
    if file_ext in ["jpg", "jpeg", "png"]:
        st.image(temp_path, caption="📷 Uploaded Image", use_container_width=True)
        results = model.predict(temp_path, verbose=False)
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="🧠 Detected Sign", use_container_width=True)

        detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
        if detected_classes:
            letter = detected_classes[0]
            st.success(f"✅ Detected Sign: **{letter}**")
            st.info(asl_meanings.get(letter, "Meaning not available."))
        else:
            st.warning("No sign detected. Try another image.")

    # --------------------------
    # If video uploaded
    # --------------------------
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
            detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
            if detected_classes:
                detected_letters.append(detected_classes[0])

        cap.release()

        if detected_letters:
            final_letter = max(set(detected_letters), key=detected_letters.count)
            st.success(f"✅ Detected Sign: **{final_letter}**")
            st.info(asl_meanings.get(final_letter, "Meaning not available."))
        else:
            st.warning("No sign detected in video. Try again with clearer lighting or background.")
