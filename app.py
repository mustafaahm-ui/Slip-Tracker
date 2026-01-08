import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from ultralytics import YOLO
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# --- إعداد الصفحة ---
st.set_page_config(page_title="Tractor Slip Analyzer", layout="wide", page_icon="🚜")

# --- تهيئة متغيرات الذاكرة ---
if 'v_theo' not in st.session_state:
    st.session_state.v_theo = 0.0
if 'ppm' not in st.session_state:
    st.session_state.ppm = 0.0  # Pixels per meter

# --- دوال مساعدة ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except:
    st.error("Model 'best.pt' not found!")
    st.stop()

# --- فئة معالجة الفيديو المباشر (WebRTC) ---
class TractorTracker(VideoTransformerBase):
    def __init__(self):
        self.ppm = st.session_state.ppm
        self.v_theo = st.session_state.v_theo
        self.mode = "calibrating" if self.v_theo == 0 else "measuring"
        self.prev_y = None
        self.last_time = time.time()
        self.dist_accumulated = 0
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        current_time = time.time()
        
        # تتبع الكائنات
        results = self.model.track(img, persist=True, verbose=False)
        
        curr_speed_kmh = 0.0
        slip_ratio = 0.0
        
        if results[0].boxes.id is not None:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            center_y = int((box[1] + box[3]) / 2)
            
            # رسم المربع
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            
            # حساب السرعة
            if self.prev_y is not None and self.ppm > 0:
                pixel_move = abs(center_y - self.prev_y)
                time_diff = current_time - self.last_time
                
                if time_diff > 0:
                    dist_m = pixel_move / self.ppm
                    speed_ms = dist_m / time_diff
                    curr_speed_kmh = speed_ms * 3.6
                    
                    # تنعيم القراءة (تجاهل القفزات غير المنطقية)
                    if curr_speed_kmh < 30: 
                        self.dist_accumulated += dist_m

            self.prev_y = center_y
            self.last_time = current_time
        
        # --- عرض المعلومات على الشاشة ---
        
        # 1. وضع المعايرة (الأسفلت)
        if self.mode == "calibrating":
            cv2.putText(img, "MODE: REFERENCE RUN (ASPHALT)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(img, f"Current Speed: {curr_speed_kmh:.1f} km/h", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # هنا نقوم فقط بعرض السرعة، والمستخدم سيأخذ القيمة يدوياً أو نطور كوداً لحفظها
            
        # 2. وضع القياس (الحقل)
        else:
            if self.v_theo > 0:
                slip_ratio = ((self.v_theo - curr_speed_kmh) / self.v_theo) * 100
            
            # ألوان الحالة
            color = (0, 255, 0)
            status = "Safe"
            if slip_ratio > 15: color, status = (0, 255, 255), "Warning"
            if slip_ratio > 20: color, status = (0, 0, 255), "Slip!"

            cv2.rectangle(img, (0, 0), (350, 150), (0, 0, 0), -1)
            cv2.putText(img, f"V_Act: {curr_speed_kmh:.1f} km/h", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f"V_Ref: {self.v_theo:.1f} km/h", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            cv2.putText(img, f"SLIP: {slip_ratio:.1f}%", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        return img

# --- واجهة المستخدم ---
st.title("🚜 Live Tractor Slip Detector")

# اختيار المصدر
source_option = st.radio("Select Input Source:", ("📂 Upload Video", "📷 Live Camera (WebRTC)"))

# --- قسم المعايرة (مبسط) ---
with st.expander("⚙️ Step 1: Calibration (Pixels per Meter)", expanded=True):
    st.write("Draw lines on screen conceptually. If distance between markers is 2m:")
    real_dist = st.number_input("Real Distance (m)", value=2.0)
    pixel_dist = st.number_input("Pixels on screen (Estimate)", value=200)
    
    if st.button("Set PPM"):
        st.session_state.ppm = pixel_dist / real_dist
        st.success(f"PPM Set: {st.session_state.ppm}")

# --- قسم السرعة المرجعية ---
with st.expander("🏎️ Step 2: Set Reference Speed (Asphalt)", expanded=True):
    col1, col2 = st.columns(2)
    manual_v = col1.number_input("Enter V_theo manually (if known)", value=5.4)
    if col1.button("Set V_theo"):
        st.session_state.v_theo = manual_v
        st.success(f"Reference Speed Fixed: {manual_v} km/h")
    
    col2.metric("Current V_theo", f"{st.session_state.v_theo} km/h")

# --- الشاشة الرئيسية ---
st.markdown("### 📺 Monitoring Screen")

if source_option == "📷 Live Camera (WebRTC)":
    st.info("Ensure you allow camera access. Works on Mobile & PC.")
    webrtc_streamer(key="tractor", video_transformer_factory=TractorTracker)

else: # Upload Video
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi'])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        st_frame = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # نفس منطق المعالجة هنا (مبسط للعرض)
            results = model.track(frame, persist=True, verbose=False)
            if results[0].boxes.id is not None:
                box = results[0].boxes.xyxy[0].cpu().numpy()
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            
            st_frame.image(frame, channels="BGR")