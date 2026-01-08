import cv2
import streamlit as st
from ultralytics import YOLO

# 1. Page Setup
st.set_page_config(page_title="AI Fitness Tracker", layout="wide")
st.title("🏋️‍♂️ AI Fitness Counter (Object Detection)")

# 2. Session State Initialization
# Keeps track of count and state between frame updates
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'state' not in st.session_state:
    st.session_state.state = "START" 

# 3. Sidebar Controls
st.sidebar.header("Settings")

# Caching the model load to prevent reloading on every interaction
@st.cache_resource
def load_model():
    # Replace 'best.pt' with your specific model file if named differently
    return YOLO('best.pt')

try:
    model = load_model()
    st.sidebar.success("Model Loaded Successfully ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Slider for line position
line_pos = st.sidebar.slider("Line Position (%)", min_value=10, max_value=90, value=50)

# Reset Button
reset_btn = st.sidebar.button("Reset Counter 🔄")

if reset_btn:
    st.session_state.counter = 0
    st.session_state.state = "START"

# 4. Main Execution
run_camera = st.checkbox('Start Camera & Tracking', value=False)
frame_placeholder = st.empty()

# Camera Setup (0 for default webcam)
cap = cv2.VideoCapture(0)

while run_camera:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not accessible.")
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Get frame dimensions
    h, w, _ = frame.shape
    
    # Calculate line Y-coordinate based on slider percentage
    line_y = int(h * (line_pos / 100))

    # =========================================================
    # 5. Tracking Logic (BoT-SORT)
    # =========================================================
    # persist=True: Maintains ID tracking across frames
    results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

    # Process detections
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            # --- Calculate Centroid (Center of the box) ---
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw center point and bounding box
            cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # --- Counting Logic ---
            # 1. Crossing DOWN (Movement Phase 1)
            if center_y > line_y: 
                if st.session_state.state == "UP": 
                    st.session_state.counter += 1
                    st.session_state.state = "DOWN"
            
            # 2. Crossing UP (Reset Phase)
            elif center_y < line_y:
                st.session_state.state = "UP"

            # Display ID above the box
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # =========================================================
    # 6. Visualization & UI Overlays
    # =========================================================
    
    # Draw Threshold Line
    line_color = (0, 0, 255) # Red
    cv2.line(frame, (0, line_y), (w, line_y), line_color, 3)

    # Create a background rectangle for text visibility
    cv2.rectangle(frame, (0, 0), (280, 90), (0, 0, 0), -1) 
    
    # Display Stats on Video
    cv2.putText(frame, f"Count: {st.session_state.counter}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    cv2.putText(frame, f"State: {st.session_state.state}", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Add simple instruction on the line
    cv2.putText(frame, "Threshold Line", (10, line_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)

    # Convert BGR to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Update Streamlit image
    frame_placeholder.image(frame_rgb)

# Release resources
cap.release()