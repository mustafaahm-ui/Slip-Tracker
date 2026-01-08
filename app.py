import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# 1. إعداد الصفحة بعنوان مناسب
st.set_page_config(page_title="Tractor Slippage Detector", layout="wide")
st.title("🚜 Tractor Slippage Detection & Counting")

# 2. تحميل الموديل
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. إعدادات الشريط الجانبي (محدثة)
st.sidebar.header("Configuration")

# خيار لتحديد اتجاه الخط (مهم جداً للجرارات)
line_orientation = st.sidebar.radio("Line Orientation:", ["Vertical (Left-Right)", "Horizontal (Up-Down)"])
line_pos = st.sidebar.slider("Line Position (%)", 10, 90, 50)

# زر التصفير
if st.sidebar.button("Reset Counter 🔄"):
    st.session_state.counter = 0

# تهيئة العداد
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'state' not in st.session_state:
    st.session_state.state = "A" # المنطقة الأولى

# 4. معالج الفيديو
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.line_pos = line_pos
        self.orientation = line_orientation

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # الحصول على الأبعاد
        h, w, _ = img.shape
        
        # إعداد الموديل للتتبع
        results = self.model.track(img, persist=True, tracker="botsort.yaml", verbose=False)

        # رسم الخط وحساب موقعه
        line_color = (0, 0, 255) # أحمر
        
        if self.orientation == "Horizontal (Up-Down)":
            line_val = int(h * (self.line_pos / 100))
            cv2.line(img, (0, line_val), (w, line_val), line_color, 2)
        else: # Vertical
            line_val = int(w * (self.line_pos / 100))
            cv2.line(img, (line_val, 0), (line_val, h), line_color, 2)

        # منطق العد
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # تحديد القيمة التي نقارن بها بناءً على اتجاه الخط
                current_val = center_y if self.orientation == "Horizontal (Up-Down)" else center_x
                
                # رسم النقطة
                cv2.circle(img, (center_x, center_y), 5, (0, 255, 255), -1)

                # --- خوارزمية العد (Counting Logic) ---
                # المنطقة A (قبل الخط) والمنطقة B (بعد الخط)
                
                if current_val > line_val:
                    if st.session_state.state == "A":
                        st.session_state.counter += 1
                        st.session_state.state = "B"
                elif current_val < line_val:
                    if st.session_state.state == "B":
                        st.session_state.state = "A" # إعادة تهيئة للعدة القادمة

                # رسم المربع والمعرف
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"ID: {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # عرض العداد على الشاشة
        cv2.rectangle(img, (0, 0), (250, 60), (0, 0, 0), -1)
        cv2.putText(img, f"Slippage Count: {st.session_state.counter}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5. تشغيل الكاميرا
st.write("Click **START** below to activate the camera detection:")
webrtc_streamer(key="tractor-tracker", video_processor_factory=VideoProcessor)

st.info("💡 Note: If counting implies wheel rotation, align the line with a fixed point on the chassis.")