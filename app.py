import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title="ASL Detection with Webcam", layout="wide")

# ------------------------------
# Load your trained YOLOv8 model
# Use the path where your trained weights are stored
# Example: after training, best.pt is in /content/runs/detect/train2/weights/best.pt
# ------------------------------
model = YOLO("best.pt")

st.title("American Sign Language Detection")
st.write("Use your webcam to capture gestures, and YOLOv8 will detect them.")

# ------------------------------
# Webcam input
# ------------------------------
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Convert to PIL Image
    image = Image.open(img_file_buffer)
    st.image(image, caption='Captured Image', use_column_width=True)

    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # ------------------------------
    # YOLOv8 prediction
    # ------------------------------
    results = model.predict(image_cv)

    # Annotate prediction
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(annotated_frame, caption='Prediction', use_column_width=True)




