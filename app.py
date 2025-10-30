import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.title("🖐️ ASL Detection (Streamlit Cloud Compatible)")

# Load YOLO model
model = YOLO("best.pt")

# Take a picture using Streamlit's camera input
img_file = st.camera_input("Take a picture")

if img_file is not None:
    image = Image.open(img_file)
   

    # Convert to OpenCV
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # YOLO prediction
    results = model.predict(image_cv)
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(annotated_frame, caption="Prediction", use_column_width=True)

    # Extract detected letter
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        class_id = int(boxes.cls[0])
        class_name = model.names[class_id]
        st.success(f"✅ Detected Letter: {class_name}")
    else:
        st.warning("⚠️ No hand sign detected. Try again!")

