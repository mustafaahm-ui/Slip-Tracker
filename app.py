import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from ultralytics import YOLO
import time

# --- إعداد الصفحة ---
st.set_page_config(page_title="Tractor Slip Analyzer", layout="wide", page_icon="🚜")

# --- تهيئة متغيرات الذاكرة (Session State) ---
if 'v_theo' not in st.session_state:
    st.session_state.v_theo = 0.0
if 'ppm_asphalt' not in st.session_state:
    st.session_state.ppm_asphalt = 0.0
if 'logs' not in st.session_state:
    st.session_state.logs = []

# --- دوال مساعدة ---
@st.cache_resource
def load_model():
    # تأكد من وجود ملف best.pt أو سيستخدم نموذجاً عاماً
    try:
        return YOLO('best.pt')
    except:
        st.warning("⚠️ 'best.pt' not found. Using 'yolov8n.pt' for testing.")
        return YOLO('yolov8n.pt')

def get_video_frame(video_path):
    """دالة لجلب أول إطار من الفيديو للمعايرة"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame.shape[1], frame.shape[0] # width, height
    return None, 0, 0

# --- واجهة التطبيق ---
st.title("🚜 Tractor Slippage Analysis System")
st.markdown("---")

# إنشاء التبويبات
tab1, tab2 = st.tabs(["🛣️ 1. Reference Run (Asphalt)", "🌾 2. Plowing Test (Field)"])

# ==========================================
# الواجهة الأولى: المعايرة على الأسفلت
# ==========================================
with tab1:
    st.header("1. Determine Theoretical Speed ($V_{Theoretical}$)")
    st.info("Upload a video of the tractor running on asphalt (no slip) to establish the baseline speed.")

    video_file_1 = st.file_uploader("Upload Asphalt Video", type=['mp4', 'avi', 'mov'], key="v1")

    if video_file_1:
        # حفظ الفيديو مؤقتاً
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file_1.read())
        path1 = tfile.name

        # --- خطوة المعايرة (PPM Calibration) ---
        st.subheader("🛠️ Calibration (Pixels Per Meter)")
        col_cal1, col_cal2 = st.columns([2, 1])
        
        frame_img, w, h = get_video_frame(path1)
        
        with col_cal2:
            real_dist = st.number_input("Known Distance on ground (meters):", value=2.0, step=0.1)
            # سلايدر لتحديد موقع الخطين
            line1_y = st.slider("Line 1 Position (Y)", 0, h, int(h*0.3))
            line2_y = st.slider("Line 2 Position (Y)", 0, h, int(h*0.7))
            
            pixel_dist = abs(line2_y - line1_y)
            ppm = pixel_dist / real_dist if real_dist > 0 else 1
            
            st.metric("Calculated PPM", f"{ppm:.2f} px/m")
            
            if st.button("Confirm Calibration ✅", key="cal_btn1"):
                st.session_state.ppm_asphalt = ppm
                st.success(f"Calibration Saved: {ppm:.2f} PPM")

        with col_cal1:
            # رسم الخطوط على الصورة للمعاينة
            if frame_img is not None:
                preview = frame_img.copy()
                cv2.line(preview, (0, line1_y), (w, line1_y), (255, 0, 0), 5) # أحمر
                cv2.line(preview, (0, line2_y), (w, line2_y), (0, 255, 0), 5) # أخضر
                st.image(preview, caption="Calibration Lines Setup", use_container_width=True)

        # --- زر التشغيل ---
        if st.button("▶️ Start Analysis (Calculate V_theo)", key="run1"):
            if st.session_state.ppm_asphalt == 0:
                st.error("Please confirm calibration first!")
            else:
                model = load_model()
                cap = cv2.VideoCapture(path1)
                fps = cap.get(cv2.CAP_PROP_FPS)
                st_frame = st.empty()
                
                speeds = []
                prev_y = None
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # تتبع الجرار
                    results = model.track(frame, persist=True, verbose=False)
                    
                    curr_speed = 0.0
                    
                    if results[0].boxes.id is not None:
                        # نفترض أننا نتبع أول كائن (الجرار)
                        box = results[0].boxes.xyxy[0].cpu().numpy()
                        center_y = int((box[1] + box[3]) / 2)
                        
                        # حساب السرعة
                        if prev_y is not None:
                            pixel_move = abs(center_y - prev_y)
                            dist_m = pixel_move / st.session_state.ppm_asphalt
                            speed_ms = dist_m * fps
                            curr_speed = speed_ms * 3.6 # تحويل لكم/ساعة
                            speeds.append(curr_speed)
                        
                        prev_y = center_y
                        
                        # رسم
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                    # عرض الفيديو
                    cv2.putText(frame, f"Speed: {curr_speed:.2f} km/h", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    st_frame.image(frame, channels="BGR")
                
                cap.release()
                
                # النتائج النهائية للتبويب الأول
                if len(speeds) > 0:
                    avg_speed = sum(speeds) / len(speeds)
                    max_speed = max(speeds)
                    
                    # نستخدم السرعة القصوى المستقرة كمرجع (أو المتوسط حسب الرغبة)
                    final_v_theo = max_speed 
                    
                    st.success(f"Analysis Complete!")
                    st.metric("Calculated Theoretical Speed ($V_{theo}$)", f"{final_v_theo:.2f} km/h")
                    
                    if st.button("Set as Reference Speed 🔒"):
                        st.session_state.v_theo = final_v_theo
                        st.toast("Reference Speed Saved! Go to Tab 2.", icon="✅")


# ==========================================
# الواجهة الثانية: اختبار الحراثة
# ==========================================
with tab2:
    st.header("2. Field Plowing Test & Slip Measurement")
    
    # التحقق من وجود السرعة المرجعية
    if st.session_state.v_theo == 0:
        st.warning("⚠️ Please complete Step 1 (Asphalt Run) first to determine V_theoretical.")
    else:
        # أشرطة المعلومات العلوية
        c1, c2, c3 = st.columns(3)
        c1.metric("Reference Speed ($V_{theo}$)", f"{st.session_state.v_theo:.2f} km/h", delta_color="off")
        
        # مدخلات المستخدم
        depth = c2.number_input("Plowing Depth (cm)", value=25)
        run_length = c3.number_input("Test Length (m)", value=50)
        
        video_file_2 = st.file_uploader("Upload Field Video", type=['mp4', 'avi', 'mov'], key="v2")
        
        if video_file_2:
            tfile2 = tempfile.NamedTemporaryFile(delete=False)
            tfile2.write(video_file_2.read())
            path2 = tfile2.name
            
            # --- معايرة جديدة للحقل (لأن العجلة تغوص) ---
            with st.expander("🛠️ Re-Calibrate for Field (Important)", expanded=True):
                col_f1, col_f2 = st.columns([2, 1])
                frame_img2, w2, h2 = get_video_frame(path2)
                
                with col_f2:
                    real_dist_f = st.number_input("Field Marker Dist (m):", value=2.0)
                    line1_yf = st.slider("Line 1 (Y)", 0, h2, int(h2*0.3), key="f1")
                    line2_yf = st.slider("Line 2 (Y)", 0, h2, int(h2*0.7), key="f2")
                    ppm_f = abs(line2_yf - line1_yf) / real_dist_f
                    st.write(f"Field PPM: **{ppm_f:.2f}**")
                
                with col_f1:
                     if frame_img2 is not None:
                        preview2 = frame_img2.copy()
                        cv2.line(preview2, (0, line1_yf), (w2, line1_yf), (255, 0, 0), 5)
                        cv2.line(preview2, (0, line2_yf), (w2, line2_yf), (0, 255, 0), 5)
                        st.image(preview2, use_container_width=True)

            # --- بدء الاختبار ---
            if st.button("▶️ START FIELD TEST", type="primary"):
                model = load_model()
                cap = cv2.VideoCapture(path2)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # أماكن العرض
                dashboard = st.columns(4)
                chart_place = st.empty()
                video_place = st.empty()
                
                # لتخزين البيانات
                df_data = []
                total_dist = 0
                prev_y = None
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    results = model.track(frame, persist=True, verbose=False)
                    
                    v_act = 0.0
                    slip_ratio = 0.0
                    status = "Safe"
                    color_status = (0, 255, 0) # Green
                    
                    if results[0].boxes.id is not None:
                        box = results[0].boxes.xyxy[0].cpu().numpy()
                        center_y = int((box[1] + box[3]) / 2)
                        
                        if prev_y is not None:
                            pixel_move = abs(center_y - prev_y)
                            dist_m = pixel_move / ppm_f
                            total_dist += dist_m
                            
                            speed_ms = dist_m * fps
                            v_act = speed_ms * 3.6
                            
                            # معادلة الانزلاق
                            if st.session_state.v_theo > 0:
                                slip_ratio = ((st.session_state.v_theo - v_act) / st.session_state.v_theo) * 100
                            
                            # منطق التنبيهات (Traffic Light Logic)
                            if slip_ratio <= 15:
                                status = "Safe 🟢"
                                color_status = (0, 255, 0)
                            elif 15 < slip_ratio <= 20:
                                status = "Warning 🟡"
                                color_status = (0, 255, 255)
                            else: # > 20
                                status = "Excessive 🔴"
                                color_status = (0, 0, 255)

                            # تسجيل البيانات
                            df_data.append({
                                "Time": cap.get(cv2.CAP_PROP_POS_MSEC)/1000,
                                "Distance (m)": total_dist,
                                "Actual Speed (km/h)": v_act,
                                "Slip Ratio (%)": slip_ratio,
                                "Status": status,
                                "Depth (cm)": depth
                            })
                        
                        prev_y = center_y
                        
                        # رسم على الفيديو
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_status, 3)
                        
                    # تحديث لوحة التحكم (Dashboard)
                    with dashboard[0]: st.metric("Actual Speed", f"{v_act:.1f} km/h")
                    with dashboard[1]: st.metric("Slip Ratio", f"{slip_ratio:.1f} %")
                    with dashboard[2]: st.metric("Distance", f"{total_dist:.1f} m")
                    with dashboard[3]: st.markdown(f"## {status}")

                    # عرض الفيديو
                    video_place.image(frame, channels="BGR")
                    
                    # تحديث الرسم البياني (كل 5 إطارات لتخفيف الضغط)
                    if len(df_data) > 0 and len(df_data) % 5 == 0:
                        chart_df = pd.DataFrame(df_data)
                        chart_place.line_chart(chart_df[["Actual Speed (km/h)", "Slip Ratio (%)"]])

                cap.release()
                
                # --- تصدير البيانات ---
                st.success("Test Completed!")
                if len(df_data) > 0:
                    final_df = pd.DataFrame(df_data)
                    st.dataframe(final_df.head())
                    
                    csv = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Data Report (CSV)",
                        data=csv,
                        file_name='tractor_slip_report.csv',
                        mime='text/csv',
                    )