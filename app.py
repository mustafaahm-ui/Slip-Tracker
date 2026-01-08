import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# --- إعداد الصفحة ---
st.set_page_config(page_title="Live Tractor Speed Trap", layout="wide", page_icon="🚜")

# --- تهيئة الذاكرة (Session State) ---
# المتغيرات التي نريد حفظها بين التبويبات
if 'v_theo' not in st.session_state:
    st.session_state.v_theo = 0.0
if 'last_measured_speed' not in st.session_state:
    st.session_state.last_measured_speed = 0.0

# القيم الافتراضية للخطوط (لتعديلها من الواجهة)
if 'line1_percent' not in st.session_state: st.session_state.line1_percent = 20
if 'line2_percent' not in st.session_state: st.session_state.line2_percent = 80
if 'trap_distance' not in st.session_state: st.session_state.trap_distance = 20.0
if 'reset_trigger' not in st.session_state: st.session_state.reset_trigger = False

# --- تحميل الموديل ---
@st.cache_resource
def load_model():
    try:
        return YOLO('best.pt')
    except:
        return YOLO('yolov8n.pt')

model = load_model()

# --- كلاس معالجة الفيديو (المخ المنفذ) ---
class SpeedTrapProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.start_time = None
        self.end_time = None
        self.measured_speed = 0.0
        self.state = "WAITING" # WAITING -> RUNNING -> FINISHED
        
        # لقراءة الإعدادات الحالية من الواجهة
        self.l1_pct = st.session_state.get('line1_percent', 20)
        self.l2_pct = st.session_state.get('line2_percent', 80)
        self.dist = st.session_state.get('trap_distance', 20.0)
        
    def transform(self, frame):
        # التحقق من طلب إعادة الضبط (Reset)
        if st.session_state.get('reset_trigger', False):
            self.start_time = None
            self.end_time = None
            self.measured_speed = 0.0
            self.state = "WAITING"
            # لا نستطيع تغيير session_state هنا، لذا سنعتمد على أن الزر في الواجهة سيغيره
        
        # تحديث مواقع الخطوط ديناميكياً إذا غيرها المستخدم
        self.l1_pct = st.session_state.get('line1_percent', 20)
        self.l2_pct = st.session_state.get('line2_percent', 80)
        self.dist = st.session_state.get('trap_distance', 20.0)

        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # تحويل النسب إلى بكسل
        x1 = int(w * (self.l1_pct / 100))
        x2 = int(w * (self.l2_pct / 100))
        
        # تتبع الجرار
        results = self.model.track(img, persist=True, verbose=False)
        
        tractor_x = 0
        detected = False
        
        if results[0].boxes.id is not None:
            # نأخذ أول كائن (الجرار)
            box = results[0].boxes.xyxy[0].cpu().numpy()
            # نستخدم مقدمة الجرار (أقصى اليمين للمربع إذا كان يتجه لليمين)
            # أو المركز ليكون أدق
            tractor_x = int((box[0] + box[2]) / 2)
            detected = True
            
            # رسم الجرار
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
            cv2.circle(img, (tractor_x, int((box[1]+box[3])/2)), 8, (0, 0, 255), -1)

        # --- منطق القياس (Timing Logic) ---
        current_t = time.time()
        
        if self.state == "WAITING":
            if detected and tractor_x > x1:
                self.start_time = current_t
                self.state = "RUNNING"
                
        elif self.state == "RUNNING":
            if detected and tractor_x > x2:
                self.end_time = current_t
                self.state = "FINISHED"
                
                # حساب السرعة فوراً
                duration = self.end_time - self.start_time
                if duration > 0:
                    speed_ms = self.dist / duration
                    self.measured_speed = speed_ms * 3.6 # كم/ساعة
                    
                    # حفظ النتيجة في مكان يمكن للواجهة قراءته (خدعة الـ Queue ممكنة لكن هنا سنعرضها فقط)
        
        # --- الرسم على الشاشة ---
        # الخط الأول (Start)
        color_l1 = (0, 255, 0) if self.state == "WAITING" else (100, 100, 100)
        cv2.line(img, (x1, 0), (x1, h), color_l1, 2)
        cv2.putText(img, "START", (x1+5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_l1, 2)
        
        # الخط الثاني (Finish)
        color_l2 = (0, 0, 255)
        cv2.line(img, (x2, 0), (x2, h), color_l2, 2)
        cv2.putText(img, "FINISH", (x2+5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_l2, 2)
        
        # عرض الحالة والسرعة
        cv2.rectangle(img, (0, 0), (w, 80), (0, 0, 0), -1) # شريط علوي أسود
        
        if self.state == "WAITING":
            status_text = "READY: Drive Tractor ->"
            cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        elif self.state == "RUNNING":
            elapsed = current_t - self.start_time
            status_text = f"MEASURING... Time: {elapsed:.2f} s"
            cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
        elif self.state == "FINISHED":
            res_text = f"DONE! Speed: {self.measured_speed:.2f} km/h"
            cv2.putText(img, res_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        return img

# --- واجهة المستخدم ---
st.title("🚜 Live Timing Gate (Camera)")

# أزرار التحكم العلوية
col_conf1, col_conf2, col_conf3 = st.columns([1, 2, 1])
with col_conf1:
    dist_input = st.number_input("Trap Distance (m)", value=20.0, step=1.0)
    st.session_state.trap_distance = dist_input
with col_conf2:
    # أزرار تحديد المواقع للخطوط
    l1 = st.slider("Start Line Position (Left)", 0, 100, 20)
    l2 = st.slider("Finish Line Position (Right)", 0, 100, 80)
    st.session_state.line1_percent = l1
    st.session_state.line2_percent = l2
with col_conf3:
    if st.button("🔄 RESET SYSTEM", type="primary"):
        st.session_state.reset_trigger = True
        # خدعة بسيطة لإعادة التفعيل سريعاً
        time.sleep(0.1)
        st.session_state.reset_trigger = False
        st.rerun()

st.markdown("---")

# التبويبات
tab1, tab2 = st.tabs(["🛣️ 1. Asphalt (Reference)", "🌾 2. Field (Measurement)"])

with tab1:
    st.markdown("### Step 1: Measure Theoretical Speed")
    st.info("Align the lines with your markers on the ground. Drive the tractor through.")
    
    # تشغيل الكاميرا
    ctx1 = webrtc_streamer(key="trap-cam-1", video_transformer_factory=SpeedTrapProcessor, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    # مكان لإدخال السرعة يدوياً بعد رؤيتها على الشاشة
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        manual_speed = st.number_input("Enter the Speed shown on camera (km/h):", value=0.0, step=0.1)
    with col_res2:
        if st.button("💾 Save as Theoretical Speed"):
            st.session_state.v_theo = manual_speed
            st.success(f"Saved: {manual_speed} km/h")

with tab2:
    st.markdown("### Step 2: Measure Slip")
    
    if st.session_state.v_theo == 0:
        st.warning("Please set Theoretical Speed in Tab 1 first.")
    else:
        st.metric("Theoretical Speed (Fixed)", f"{st.session_state.v_theo} km/h")
        
        # تشغيل الكاميرا للحقل
        ctx2 = webrtc_streamer(key="trap-cam-2", video_transformer_factory=SpeedTrapProcessor, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        
        # حاسبة الانزلاق
        st.markdown("#### Slip Calculator")
        c1, c2, c3 = st.columns(3)
        field_speed = c1.number_input("Enter Field Speed (from camera):", min_value=0.0)
        
        if field_speed > 0:
            slip = ((st.session_state.v_theo - field_speed) / st.session_state.v_theo) * 100
            c2.metric("Calculated Slip", f"{slip:.2f} %")
            
            status = "Unknown"
            if slip < 15: status = "✅ Good"
            elif slip < 20: status = "⚠️ Warning"
            else: status = "🔴 HIGH SLIP"
            c3.markdown(f"## {status}")