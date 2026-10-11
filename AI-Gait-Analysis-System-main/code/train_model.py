import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Load CSV
df = pd.read_csv("C:\\Users\\raksh\\Downloads\\Gait_Analysis_project\\dataset\\gait_features.csv")

# Encode labels
df['label'] = df['label'].map({"normal": 1, "abnormal": 0})

X = df[['avg_knee_angle', 'avg_hip_angle', 'steps', 'cadence']]
y = df['label']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# ML model
model = SVC(kernel='rbf', probability=True)

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("C:\\Users\\raksh\\Downloads\\Gait_Analysis_project\\models\\gait_model.pkl", "wb"))

print("Model trained and saved as gait_model.pkl ✔")
