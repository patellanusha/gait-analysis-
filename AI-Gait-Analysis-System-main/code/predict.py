import cv2
import mediapipe as mp
import numpy as np
import pickle
import math

# Load trained ML model
model = pickle.load(open("../models/gait_model.pkl", "rb"))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ---------- UTILITY : ANGLE CALCULATION ----------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_value = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cos_value))

# ----------- VIDEO PATH ---------------
cap = cv2.VideoCapture("../videos/test_walk.mp4")  # Change if required

knee_angles = []
hip_angles = []
steps = 0
frame_count = 0

# Both legs
prev_left_y = None
prev_right_y = None
STEP_THRESHOLD = 0.015

print("\n🟢 Analyzing gait, please wait...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Key Joints
        left_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
        left_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
        left_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]

        right_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
        right_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        right_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

        right_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

        # Knee & Hip angles
        knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        knee_angles.append(knee_angle)
        hip_angles.append(hip_angle)

        # Step detection
        left_y = left_ankle[1]
        right_y = right_ankle[1]

        if prev_left_y is not None and prev_right_y is not None:
            if abs(left_y - prev_left_y) > STEP_THRESHOLD or abs(right_y - prev_right_y) > STEP_THRESHOLD:
                steps += 1

        prev_left_y = left_y
        prev_right_y = right_y

cap.release()

# Final gait metrics
avg_knee = np.mean(knee_angles)
avg_hip = np.mean(hip_angles)

fps = 30
cadence = steps / (frame_count / fps) * 60 if frame_count != 0 else 0

print("\n=========== Extracted Features ===========")
print(f"Knee   : {avg_knee:.2f}")
print(f"Hip    : {avg_hip:.2f}")
print(f"Steps  : {steps}")
print(f"Cadence: {cadence:.2f}")
print("=========================================\n")

# --------- FEATURE INPUT TO MODEL ---------
features = np.array([[avg_knee, avg_hip, steps, cadence]])

prediction = model.predict(features)[0]

# Convert numeric → label
if prediction == 1:
    result = "NORMAL"
else:
    result = "ABNORMAL"

print("👉 Final Prediction:", result.upper())
print("=========================================")
