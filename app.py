import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(page_title="ASL Sign Language Translator", layout="wide")
st.title("🤟 American Sign Language (ASL) → Text Translator")
st.write("Upload an image or use your webcam to detect hand signs and translate them into letters.")

# ---------------------- Load YOLO Model ----------------------
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please upload your trained YOLO model as `best.pt` in the same directory.")
    st.stop()

model = YOLO(MODEL_PATH)
st.success("✅ YOLO model loaded successfully!")

# ---------------------- App Mode Selection ----------------------
mode = st.radio("Select Mode:", ["📷 Image Upload", "🎥 Live Webcam"], horizontal=True)

# ---------------------- IMAGE UPLOAD MODE ----------------------
if mode == "📷 Image Upload":
    uploaded_file = st.file_uploader("Upload an Image of a Hand Sign", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Read image
        img = Image.open(uploaded_file)
        img_array = np.array(img)

        # Run detection
        results = model.predict(img_array, verbose=False)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Extract detected sign
        detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
        if detected_classes:
            st.subheader(f"🧠 Detected Sign: **{detected_classes[0]}**")
        else:
            st.warning("⚠️ No sign detected in this image.")

        # Display annotated image
        st.image(annotated_frame, caption="Detected Result", use_column_width=True)

# ---------------------- WEBCAM MODE ----------------------
elif mode == "🎥 Live Webcam":
    st.info("Click the checkbox below to start your webcam and show ASL hand signs!")

    run = st.checkbox("✅ Start Webcam")
    FRAME_WINDOW = st.image([])
    detected_text = ""

    if run:
        cap = cv2.VideoCapture(0)
        st.write("📹 Webcam started — show your ASL signs!")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Could not access webcam.")
                break

            # Run YOLO prediction
            results = model.predict(frame, verbose=False)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Get detected classes (signs)
            detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
            if detected_classes:
                letter = detected_classes[0]
                # Avoid repeating same letter continuously
                if len(detected_text) == 0 or detected_text[-1] != letter:
                    detected_text += letter

            # Display frame and detected text
            FRAME_WINDOW.image(annotated_frame, channels="RGB")
            st.text_area("📝 Detected Alphabets:", detected_text, height=120)

            # Stop button
            stop = st.button("🛑 Stop Webcam")
            if stop:
                break

        cap.release()
        st.success("✅ Webcam stopped.")
    else:
        st.info("👆 Turn on the checkbox to start webcam.")

st.caption("Developed by Sehrish Tariq 💻 | YOLOv8-powered ASL Detection")

