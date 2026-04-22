from google.cloud import aiplatform
from google.cloud import storage
import json

# Initialize Vertex AI
aiplatform.init(
project="project-353e1991-8ed1-4de9-a96",
location="asia-south1"
)

# Load metrics from GCS (or local if running in same step)
client = storage.Client()
bucket = client.bucket("credit-card-fraud-gcp")
blob = bucket.blob("metrics/metrics.json")

metrics = json.loads(blob.download_as_text())

# Thresholds
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

        if value < threshold:
           print(f"❌ Test failed: {metric}={value:.4f} < {threshold}")
           return False

    return True


# Run validation
if tests_pass(metrics, test_thresholds):
    print("✅ Model passed validation. Registering in Vertex AI...")

    model = aiplatform.Model.upload(
        display_name="credit-card-fraud-model",
        artifact_uri="gs://credit-card-fraud-gcp/sklearn_model",
        serving_container_image_uri="us-docker.pkg.dev/project-353e1991-8ed1-4de9-a96/vertex/sklearn-predictor:latest",
        labels={
            "stage": "staging",
            "type": "challenger"
        }
    )

    print(f"🚀 Model registered: {model.resource_name}")

else:
    print("❌ Model failed validation. NOT registering.")

