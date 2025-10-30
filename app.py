import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile

st.set_page_config(page_title="YOLO Object Detection", layout="centered")

st.title("🧠 YOLO Object Detection App")
st.write("Upload an image and let the YOLO model detect objects for you!")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # smallest pre-trained YOLO model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        tmp_path = tmp_file.name

    # Run YOLO prediction
    st.write("🔍 Running YOLO model...")
    results = model.predict(source=tmp_path, conf=0.3)

    # Get result image
    result_img = results[0].plot()
    st.image(result_img, caption="Detection Result", use_column_width=True)
else:
    st.info("Please upload an image to start detection.")





