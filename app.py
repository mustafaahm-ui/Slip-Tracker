import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# --- إعداد الصفحة ---
st.set_page_config(page_title="Live Tractor Speed Trap", layout="wide", page_icon="🚜")

# --- تهيئة الذاكرة ---
if 'v_theo' not in st.session_state: st.session_state.v_theo = 0.0
if 'trap_distance' not in st.session_state: st.session_state.trap_distance = 20.0
if 'line1_percent' not in st.session_state: st.session_state.line1_percent = 20
if 'line2_percent' not in st.session_state: st.session_state.line2_percent = 80
if 'reset_trigger' not in st.session_state: st.session_state.reset_trigger = False

# --- إعدادات الاتصال (مهم جداً لحل مشكلة التوقف) ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- تحميل الموديل ---
@st.cache_resource
def load_model():
    try:
        # تحميل الموديل الأخف لضمان السرعة (يمكنك تغييره لـ best.pt لاحقاً)
        return YOLO('yolov8n.pt') 
    except:
        return None

model = load_model()

# --- معالج الفيديو ---
class SpeedTrapProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.start_time = None
        self.end_time = None
        self.measured_speed = 0.0
        self.state = "WAITING"
        self.frame_count = 0 # عداد لتخفيف الضغط
        
        # قراءة الإعدادات
        self.l1_pct = st.session_state.get('line1_percent', 20)
        self.l2_pct = st.session_state.get('line2_percent', 80)
        self.dist = st.session_state.get('trap_distance', 20.0)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. تصغير الصورة لتسريع المعالجة (حل لمشكلة التجميد)
        # نقوم بتصغيرها للتحليل فقط، لكن نعرضها بالحجم الأصلي أو أصغر قليلاً
        h_orig, w_orig, _ = img.shape
        img_resized = cv2.resize(img, (640, 480)) # حجم خفيف جداً للمعالجة
        
        # 2. تخطي الإطارات (Frame Skipping)
        # نقوم بالتحليل مرة واحدة كل 3 إطارات لتخفيف الحمل على المعالج
        self.frame_count += 1
        
        # القيم الافتراضية للرسم
        x1 = int(w_orig * (self.l1_pct / 100))
        x2 = int(w_orig * (self.l2_pct / 100))
        tractor_x = 0
        detected = False

        # --- التحليل الذكي (كل 3 إطارات فقط) ---
        if self.frame_count % 3 == 0:
            if self.model:
                results = self.model.track(img_resized, persist=True, verbose=False)
                if results[0].boxes.id is not None:
                    box = results[0].boxes.xyxy[0].cpu().numpy()
                    
                    # تحويل الإحداثيات من الحجم الصغير (640x480) إلى الحجم الأصلي
                    scale_x = w_orig / 640
                    scale_y = h_orig / 480
                    
                    x1_box = int(box[0] * scale_x)
                    y1_box = int(box[1] * scale_y)
                    x2_box = int(box[2] * scale_x)
                    y2_box = int(box[3] * scale_y)
                    
                    tractor_x = int((x1_box + x2_box) / 2)
                    detected = True
                    
                    # رسم المربع على الصورة الأصلية
                    cv2.rectangle(img, (x1_box, y1_box), (x2_box, y2_box), (0, 255, 255), 2)
                    cv2.circle(img, (tractor_x, int((y1_box+y2_box)/2)), 8, (0, 0, 255), -1)

        # إعادة الضبط اليدوي
        if st.session_state.get('reset_trigger', False):
            self.start_time = None
            self.end_time = None
            self.state = "WAITING"

        # --- منطق الوقت (يعتمد على آخر موقع معروف) ---
        current_t = time.time()
        
        if self.state == "WAITING":
            if detected and tractor_x > x1: # فرضنا الحركة من اليسار لليمين
                self.start_time = current_t
                self.state = "RUNNING"
                
        elif self.state == "RUNNING":
            if detected and tractor_x > x2:
                self.end_time = current_t
                self.state = "FINISHED"
                duration = self.end_time - self.start_time
                if duration > 0.1: # لتجنب الأخطاء
                    speed_ms = self.dist / duration
                    self.measured_speed = speed_ms * 3.6

        # --- الرسم الثابت (يظهر في كل إطار) ---
        # الخط 1
        cv2.line(img, (x1, 0), (x1, h_orig), (0, 255, 0), 2)
        cv2.putText(img, "START", (x1, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # الخط 2
        cv2.line(img, (x2, 0), (x2, h_orig), (0, 0, 255), 2)
        cv2.putText(img, "FINISH", (x2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # لوحة المعلومات
        status_text = f"State: {self.state}"
        if self.state == "FINISHED":
            status_text += f" | Speed: {self.measured_speed:.2f} km/h"
            
        cv2.rectangle(img, (0, 0), (600, 60), (0, 0, 0), -1)
        cv2.putText(img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img

# --- الواجهة ---
st.title("🚜 Live Speed Trap (Optimized)")

# التحكم
c1, c2, c3 = st.columns([1, 2, 1])
st.session_state.trap_distance = c1.number_input("Distance (m)", 20.0)
st.session_state.line1_percent = c2.slider("Start Line", 5, 45, 20)
st.session_state.line2_percent = c2.slider("Finish Line", 55, 95, 80)

if c3.button("Reset System"):
    st.session_state.reset_trigger = True
    time.sleep(0.1)
    st.session_state.reset_trigger = False
    st.rerun()

# التبويبات
t1, t2 = st.tabs(["1. Asphalt (Theo)", "2. Field (Slip)"])

with t1:
    st.write("Measure Theoretical Speed:")
    # إضافة media_stream_constraints لطلب جودة منخفضة لزيادة السرعة
    webrtc_streamer(
        key="cam1", 
        video_transformer_factory=SpeedTrapProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False}
    )
    
    manual_v = st.number_input("Recorded Speed (km/h):", 0.0)
    if st.button("Set Theoretical"):
        st.session_state.v_theo = manual_v
        st.success(f"Saved: {manual_v}")

with t2:
    if st.session_state.v_theo == 0:
        st.error("Go to Tab 1 first.")
    else:
        st.write(f"Reference Speed: **{st.session_state.v_theo} km/h**")
        webrtc_streamer(
            key="cam2", 
            video_transformer_factory=SpeedTrapProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False}
        )
        
        act_v = st.number_input("Field Speed (km/h):", 0.0)
        if act_v > 0:
            slip = ((st.session_state.v_theo - act_v)/st.session_state.v_theo)*100
            st.metric("Slip %", f"{slip:.1f}%")