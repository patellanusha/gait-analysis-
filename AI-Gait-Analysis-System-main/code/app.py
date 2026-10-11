import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import tempfile
import matplotlib.pyplot as plt

# Load ML model
model = pickle.load(open("C:\\Users\\raksh\\Downloads\\Gait_Analysis_project\\models\\gait_model.pkl", "rb"))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Angle function
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cos_val))


def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)

    knee_angles = []
    hip_angles = []
    steps = 0
    frame_count = 0
    prev_left_y = None
    prev_right_y = None
    STEP_THRESHOLD = 0.015

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:

            # ----- SKELETON OVERLAY -----
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            stframe.image(frame, channels="BGR", use_container_width=True)

            lm = results.pose_landmarks.landmark

            left_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]

            right_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
            right_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            right_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

            right_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

            knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

            knee_angles.append(knee_angle)
            hip_angles.append(hip_angle)

            left_y = left_ankle[1]
            right_y = right_ankle[1]

            if prev_left_y and prev_right_y:
                if abs(left_y - prev_left_y) > STEP_THRESHOLD or abs(right_y - prev_right_y) > STEP_THRESHOLD:
                    steps += 1

            prev_left_y = left_y
            prev_right_y = right_y

    cap.release()

    avg_knee = np.mean(knee_angles)
    avg_hip = np.mean(hip_angles)
    fps = 30
    cadence = steps / (frame_count / fps) * 60

    return avg_knee, avg_hip, steps, cadence, knee_angles, hip_angles


# ================== STREAMLIT UI ==================
st.title("🦵 AI Gait Analysis System")
st.subheader("Classifies Normal, Abnormal, Parkinson, and Stroke Gait")


uploaded_file = st.file_uploader("📤 Upload Walking Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save temporary video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.video(uploaded_file,format="video/mp4")

    if st.button("🔍 Analyze Gait"):
        with st.spinner("Extracting gait features... ⏳"):
            knee, hip, steps, cadence, K_series, H_series = extract_features(tfile.name)

        st.success("Feature Extraction Completed!")

        st.write("### 📊 Extracted Features")
        st.write(f"**Average Knee Angle:** `{knee:.2f}`")
        st.write(f"**Average Hip Angle:** `{hip:.2f}`")
        st.write(f"**Steps Detected:** `{steps}`")
        st.write(f"**Cadence:** `{cadence:.2f} steps/min`")

        features = np.array([[knee, hip, steps, cadence]])
        prediction = model.predict(features)[0]

        labels = {1: "NORMAL", 0: "ABNORMAL", 2: "PARKINSON", 3: "STROKE"}
        final = labels[int(prediction)]

        st.header(f"🟢 Result: **{final}**")

        st.write("---")
        st.subheader("📈 Joint Angle Graph")

        fig, ax = plt.subplots()
        ax.plot(K_series, label="Knee Angle")
        ax.plot(H_series, label="Hip Angle")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (degrees)")
        ax.legend()
        st.pyplot(fig)