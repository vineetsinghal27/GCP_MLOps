"""
GCP-compatible training script
Reads data from cloud storage path
Writes model + metrics to cloud storage output path
"""

import os
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from google.cloud import storage

client = storage.Client()
bucket = client.get_bucket("credit-card-fraud-gcp")
blob = bucket.blob("creditcard_train.csv")
blob.download_to_filename("creditcard_train.csv")


# -----------------------------
# Load data
# -----------------------------
data = pd.read_csv("creditcard_train.csv")

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
# Save model artifacts
# -----------------------------
os.makedirs("outputs", exist_ok=True)

model_path = "outputs/model.pkl"
metrics_path = "outputs/metrics.json"


joblib.dump(model, model_path, protocol=4)

with open(metrics_path, "w") as f:
    json.dump(metrics, f)

print("✅ Model and metrics saved")

#upload to GCS

bucket = client.bucket("credit-card-fraud-gcp")

#upload model
model_blob = bucket.blob("model.pkl")
model_blob.upload_from_filename(model_path)

#upload metrices
metrices_blob = bucket.blob("metrics.json")
metrices_blob.upload_from_filename(metrics_path)

print("✅ Model and metrics uploaded to GCS")

