import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# -----------------------------------
# Streamlit page setup
# -----------------------------------
st.set_page_config(page_title="ASL Live Detection", layout="wide")
st.title("🖐️ Real-Time American Sign Language Detection")
st.write("Show your hand sign in front of the webcam. The model will predict the corresponding alphabet.")

# -----------------------------------
# Load YOLO model
# -----------------------------------
model = YOLO("best.pt")  # path to your trained model
st.sidebar.success("✅ YOLOv8 model loaded successfully!")

# -----------------------------------
# Define video processor
# -----------------------------------
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.predicted_letter = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO prediction
        results = model.predict(img, verbose=False)

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Extract class name if any detection found
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            class_id = int(boxes.cls[0])
            self.predicted_letter = model.names[class_id]
        else:
            self.predicted_letter = "No detection"

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# -----------------------------------
# WebRTC Configuration
# -----------------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -----------------------------------
# Run live webcam stream
# -----------------------------------
ctx = webrtc_streamer(
    key="asl-detection",
    mode="sendrecv",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=ASLProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# -----------------------------------
# Display current predicted letter
# -----------------------------------
if ctx.video_processor:
    st.markdown("### 🔠 **Detected Alphabet (Live)**")
    st.text_input("Current Letter:", value=ctx.video_processor.predicted_letter)
