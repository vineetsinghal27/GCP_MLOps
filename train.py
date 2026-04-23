
"""
GCP-compatible training script
Uses pre-built GCP container (Cloud Run / Vertex AI)
"""

import os
import json
import pandas as pd
import joblib
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Environment variables
# -----------------------------
BUCKET_NAME = os.environ.get("BUCKET_NAME")
TRAIN_BLOB_PATH = "creditcard_train.csv"
LOCAL_TRAIN_FILE = "/tmp/creditcard_train.csv"

OUTPUT_DIR = os.environ.get("AIP_MODEL_DIR", "/tmp/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Download data from GCS
# -----------------------------
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

blob = bucket.blob(TRAIN_BLOB_PATH)
blob.download_to_filename(LOCAL_TRAIN_FILE)

# -----------------------------
# Load data
# -----------------------------
data = pd.read_csv(LOCAL_TRAIN_FILE)

X = data.drop("Class", axis=1)
y = data["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(x_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

print("📊 Metrics:", metrics)

# -----------------------------
# Save artifacts
# -----------------------------
model_path = os.path.join(OUTPUT_DIR, "model.pkl")
metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")

joblib.dump(model, model_path)

with open(metrics_path, "w") as f:
    json.dump(metrics, f)

print("✅ Model and metrics saved")

# -----------------------------
# Upload back to GCS
# -----------------------------
bucket.blob("model/model.pkl").upload_from_filename(model_path)
bucket.blob("metrics/metrics.json").upload_from_filename(metrics_path)

print("✅ Model and metrics uploaded to GCS")