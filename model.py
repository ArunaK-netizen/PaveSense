import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# File Paths
POTHOLE_CSV = "pothole_data.csv"
NORMAL_ROAD_CSV = "normal_road_data.csv"
MODEL_FILE = "model.p"

# Load Data
print("🔄 Loading datasets...")
pothole_data = pd.read_csv(POTHOLE_CSV)
normal_road_data = pd.read_csv(NORMAL_ROAD_CSV)

# Ensure both datasets have the same columns
assert list(pothole_data.columns) == list(normal_road_data.columns), "❌ Column mismatch between datasets!"

# Combine datasets
print("📊 Combining datasets...")
data = pd.concat([pothole_data, normal_road_data], ignore_index=True)

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Split features (X) and target labels (y)
X = data.drop(columns=["pothole"])  # Drop target column
y = data["pothole"]  # Target: 1 (pothole) or 0 (normal road)

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Model
print("🚀 Training XGBoost model...")
classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss", random_state=42)
classifier.fit(X_train_scaled, y_train)

# Evaluate Model
y_pred = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")
print("📜 Classification Report:\n", classification_report(y_test, y_pred))

# Save Model
print("💾 Saving trained model...")
model_data = {"classifier": classifier, "scaler": scaler, "feature_names": list(X.columns)}
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model_data, f)

print(f"🎉 Model saved successfully as '{MODEL_FILE}'")
