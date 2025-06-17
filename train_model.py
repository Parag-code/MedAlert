import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load Dataset
df = pd.read_csv("Data/triage_data.csv")

# Map urgency levels
urgency_map = {1: "Non-Urgent", 2: "Urgent", 3: "Critical"}
df["Urgency"] = df["Triage_Level"].map(urgency_map)
df.drop("Triage_Level", axis=1, inplace=True)

# Encode categorical features
le_gender = LabelEncoder()
le_pain = LabelEncoder()
le_mental = LabelEncoder()
le_target = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Pain"] = le_pain.fit_transform(df["Pain"])
df["Mental_Status"] = le_mental.fit_transform(df["Mental_Status"])
df["Urgency"] = le_target.fit_transform(df["Urgency"])

# Split Data
X = df[["Age", "Gender", "Pain", "Mental_Status"]]
y = df["Urgency"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Create models directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# Save model and encoders
joblib.dump(model, "models/medalert_model.pkl")
joblib.dump(scaler, "models/medalert_scaler.pkl")
joblib.dump(le_target, "models/medalert_label_encoder.pkl")

print("✅ Model training complete and saved.") 