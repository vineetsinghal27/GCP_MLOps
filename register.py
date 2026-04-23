
from google.cloud import aiplatform
from google.cloud import storage
import os
import json

# -----------------------------
# Init
# -----------------------------
PROJECT_ID = "project-353e1991-8ed1-4de9-a96"
REGION = "asia-south1"
BUCKET_NAME = os.environ.get("BUCKET_NAME", "credit-card-fraud-gcp")

aiplatform.init(
    project=PROJECT_ID,
    location=REGION
)

# -----------------------------
# Load metrics from GCS
# -----------------------------
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob("metrics/metrics.json")

metrics = json.loads(blob.download_as_text())

# -----------------------------
# Thresholds
# -----------------------------
test_thresholds = {
    "accuracy": 0.60,
    "precision": 0.60,
    "recall": 0.60,
    "f1_score": 0.60,
}

def tests_pass(metrics, thresholds):
    for metric, threshold in thresholds.items():
        value = metrics.get(metric)

        if value is None:
            print(f"⚠️ Metric '{metric}' not found")
            return False

        value = float(value)

        if value < threshold:
            print(f"❌ Test failed: {metric}={value:.4f} < {threshold}")
            return False

    return True

# -----------------------------
# Validation gate
# -----------------------------
if tests_pass(metrics, test_thresholds):
    print("✅ Model passed validation. Registering in Vertex AI...")

    model = aiplatform.Model.upload(
        display_name="credit-card-fraud-model",
        artifact_uri=f"gs://{BUCKET_NAME}/model",
        serving_container_image_uri=
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest",
        labels={
            "stage": "staging",
            "type": "challenger"
        }
    )

    model.wait()
    print(f"🚀 Model registered: {model.resource_name}")

else:
    print("❌ Model failed validation. NOT registering.")