import os
import cv2 as cv
import numpy as np
import itertools
import joblib
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av

# 1. UI Framing Configuration
st.set_page_config(page_title="Bharatanatyam Mudra Validator", layout="wide")
st.title("🩰 End-to-End Bharatanatyam Mudra Validator")
st.markdown("---")

# 2. Global State Initialization for Dynamic Cross-Column Updates
if "closest_mudra" not in st.session_state:
    st.session_state["closest_mudra"] = None
if "is_correct" not in st.session_state:
    st.session_state["is_correct"] = True

# 3. Global Cloud Cache Loaders
@st.cache_resource
def load_ml_assets():
    model = joblib.load("best_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

try:
    ML_MODEL, LABEL_ENCODER = load_ml_assets()
except Exception as e:
    st.error(f"Failed to load ML assets: {e}")
    st.stop()

# 4. Geometric Coordinates Preprocessing Logic
def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0][0], landmark_list[0][1]
    temp_list = []
    for pt in landmark_list:
        temp_list.append(pt[0] - base_x)
        temp_list.append(pt[1] - base_y)
    
    max_value = max(list(map(abs, temp_list)))
    if max_value == 0:
        return temp_list
    return [n / max_value for n in temp_list]

# 5. WebRTC Video Processing Engine Interceptor
class MudraValidationProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.confidence_threshold = 0.85

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv.flip(image, 1)
        h, w, _ = image.shape
        
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = []
                x_coordinates = []
                y_coordinates = []
                
                for lm in hand_landmarks.landmark:
                    cx, cy = min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)
                    landmark_list.append([cx, cy])
                    x_coordinates.append(cx)
                    y_coordinates.append(cy)
                
                xmin, xmax = min(x_coordinates), max(x_coordinates)
                ymin, ymax = min(y_coordinates), max(y_coordinates)
                padding = 15
                bx1, by1 = max(0, xmin - padding), max(0, ymin - padding)
                bx2, by2 = min(w, xmax + padding), min(h, ymax + padding)
                
                processed_features = pre_process_landmark(landmark_list)
                
                # Left-Hand Mirroring Fix
                hand_label = handedness.classification[0].label
                if hand_label == "Left":
                    for i in range(0, len(processed_features), 2):
                        processed_features[i] = -processed_features[i]
                
                # ML Softmax Inference Execution
                probabilities = ML_MODEL.predict_proba([processed_features])[0]
                max_prob = np.max(probabilities)
                class_id = np.argmax(probabilities)
                raw_label = LABEL_ENCODER.inverse_transform([class_id])[0]
                
                # Update background state variables asynchronously
                st.session_state["closest_mudra"] = raw_label
                
                if max_prob >= self.confidence_threshold:
                    display_text = f"{raw_label.upper()} ({max_prob*100:.1f}%)"
                    banner_color = (0, 128, 0) # Green for correct
                    st.session_state["is_correct"] = True
                else:
                    display_text = "INCORRECT MUDRA"
                    banner_color = (0, 0, 255) # Red for incorrect
                    st.session_state["is_correct"] = False
                
                # Draw bounding box and label strip on screen
                cv.rectangle(image, (bx1, by1), (bx2, by2), banner_color, 2)
                cv.rectangle(image, (bx1, by1), (bx2, by1 - 25), banner_color, -1)
                cv.putText(image, f"{hand_label}: {display_text}", (bx1 + 5, by1 - 7),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                
                # Draw hand joint skeleton dots
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0,0,0), thickness=2)
                )
        else:
            st.session_state["closest_mudra"] = None
            st.session_state["is_correct"] = True

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# 6. Streamlit Dual-Column UI Split
col1, col2 = st.columns([2, 1]) # Left column is twice as wide as the right column

with col1:
    st.subheader("📹 Live Webcam Stream")
    webrtc_streamer(
        key="mudra-streamer",
        video_processor_factory=MudraValidationProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        async_processing=True
    )

with col2:
    st.subheader("💡 Pedagogical Assistance Guide")
    
    if st.session_state["closest_mudra"] is not None:
        closest = st.session_state["closest_mudra"]
        
        if st.session_state["is_correct"]:
            st.success(f"Excellent Form! Mudra: **{closest.upper()}**")
        else:
            # Posture is wrong, highlight the targeted mudra's master photo
            st.error("🚨 Posture Alignment Error")
            st.write(f"The system detects you are attempting to form **{closest.upper()}** but your joint placements are inaccurate.")
            st.markdown("**Correct your hand posture to match this form:**")
            
            img_path = f"images/{closest.lower()}.jpg"
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Master Reference: {closest.capitalize()}", use_column_width=True)
            else:
                st.info(f"Add a reference photo at '{img_path}' to render image previews.")
    else:
        st.info("Bring your hand into the webcam frame box to initialize the AI checking system.")