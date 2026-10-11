import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# --------------- ANGLE FUNCTION -----------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_value = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cos_value))


# --------- VIDEO PATH ---------
cap = cv2.VideoCapture("../videos/test_walk.mp4")  # Update path

knee_angles = []
hip_angles = []
steps = 0
step_frames = []   # store when (frame number) step happened

prev_left_y = None
prev_right_y = None
frame_count = 0

STEP_THRESHOLD = 0.01  # movement sensitivity

print("\n🚶 Processing video... Please wait.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Left joints
        left_hip   = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
        left_knee  = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
        left_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]

        # Right joints
        right_hip  = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
        right_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        right_ankle= [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

        right_sh = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

        # ---- store angles ----
        knee_angles.append(calculate_angle(right_hip, right_knee, right_ankle))
        hip_angles.append(calculate_angle(right_sh, right_hip, right_knee))

        # STEP detection
        left_y = left_ankle[1]
        right_y = right_ankle[1]

        if prev_left_y is not None and prev_right_y is not None:
            dy_left = abs(left_y - prev_left_y)
            dy_right = abs(right_y - prev_right_y)

            if dy_left > STEP_THRESHOLD or dy_right > STEP_THRESHOLD:
                steps += 1
                step_frames.append(frame_count)  # store when step detected

        prev_left_y = left_y
        prev_right_y = right_y

cap.release()

# ---------- Compute Metrics ----------
avg_knee = np.mean(knee_angles)
avg_hip  = np.mean(hip_angles)

fps = 30
cadence = steps / (frame_count / fps) * 60


# ================== GRAPH 1 ==================
plt.figure(figsize=(10,5))
plt.plot(knee_angles, label="Knee Angle", color="blue")
plt.plot(hip_angles, label="Hip Angle", color="red")
plt.xlabel("Frame Number")
plt.ylabel("Angle (degrees)")
plt.title("Gait Joint Angles")
plt.legend()
plt.show()

# ================== GRAPH 2 ==================
plt.figure(figsize=(10,4))
plt.scatter(step_frames, [1]*len(step_frames), color="green", marker="x", s=80)
plt.yticks([])
plt.xlabel("Frame Number")
plt.title("Detected Step Events")
plt.show()


# -------- PRINT RESULTS ----------
print("\n=========== Gait Feature Output ===========")
print(f"Average Knee Angle (deg) : {avg_knee:.2f}")
print(f"Average Hip Angle  (deg) : {avg_hip:.2f}")
print(f"Steps Detected           : {steps}")
print(f"Cadence (steps/min)      : {cadence:.2f}")
print("===========================================\n")
