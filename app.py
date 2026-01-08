import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# 1. Page Config
st.set_page_config(page_title="AI Fitness Tracker", layout="wide")
st.title("🏋️‍♂️ AI Fitness Counter (Streamlit Cloud)")

# 2. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Sidebar Settings
st.sidebar.header("Settings")
line_pos = st.sidebar.slider("Line Position (%)", 10, 90, 50)

# Initialize Session State
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'state' not in st.session_state:
    st.session_state.state = "START"

# 4. Define Video Processor
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.line_pos = line_pos

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror image
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        line_y = int(h * (line_pos / 100))

        # Tracking
        results = self.model.track(img, persist=True, tracker="botsort.yaml", verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                center_y = int((y1 + y2) / 2)
                center_x = int((x1 + x2) / 2)

                # Draw Point
                cv2.circle(img, (center_x, center_y), 5, (0, 255, 255), -1)

                # Counting Logic
                # Note: We can't easily write to st.session_state from inside this thread
                # So we just visualize for now. To sync counter, simpler logic is needed.
                # Visual Feedback:
                color = (0, 255, 0)
                if center_y > line_y:
                    color = (0, 0, 255) # Crossed Down
                    cv2.putText(img, "DOWN", (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                else:
                    cv2.putText(img, "UP", (center_x, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Draw Line
        cv2.line(img, (0, line_y), (w, line_y), (255, 0, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5. Run WebRTC
st.write("Click 'START' to use camera:")
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

st.info("Note: Ensure you allow camera access in your browser.")