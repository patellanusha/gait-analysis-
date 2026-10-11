import pandas as pd
import numpy as np

# === your extracted values (from feature_extraction output) ===
avg_knee_angle = float(input("Enter Average Knee Angle: "))
avg_hip_angle = float(input("Enter Average Hip Angle : "))
steps = int(input("Enter Steps detected      : "))
cadence = float(input("Enter Cadence (steps/min) : "))

# === label the sample ===
print("\nLabel options:")
print("1 → normal")
print("2 → abnormal")

label_choice = input("Enter label number (1/2): ")

if label_choice == "1":
    label = "normal"
else:
    label = "abnormal"

# === Create dataframe row ===
data = {
    "avg_knee_angle": [avg_knee_angle],
    "avg_hip_angle": [avg_hip_angle],
    "steps": [steps],
    "cadence": [cadence],
    "label": [label]
}
df_new = pd.DataFrame(data)

# === Save / Append ===
dataset_path = (r"C:\\Users\\raksh\\Downloads\\Gait_Analysis_project\\dataset\\gait_features.csv")

try:
    # If file exists → append
    df_existing = pd.read_csv(dataset_path)
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all.to_csv(dataset_path, index=False)
    print("\n💾 Data appended to gait_features.csv")

except:
    # If file does not exist → create
    df_new.to_csv(dataset_path, index=False)
    print("\n🆕 File created: gait_features.csv")

print("\n📌 Saved entry:")
print(df_new)
