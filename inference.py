from google.cloud import aiplatform

# --------------------------------------------------
# Initialize Vertex AI
# --------------------------------------------------
aiplatform.init(
    project="project-353e1991-8ed1-4de9-a96",
    location="asia-south1"
)

# --------------------------------------------------
# Vertex AI Endpoint
# --------------------------------------------------
endpoint = aiplatform.Endpoint("4167626257317494784")

# --------------------------------------------------
# Input instance (EXACT feature order)
# --------------------------------------------------
instance = [
    1,
    -1.35981, -0.07278, 2.536347, 1.378155, -0.33832,
    0.462388, 0.239599, 0.098698, 0.363787, 0.090794,
    -0.5516, -0.6178, -0.99139, -0.31117,
    1.468177, -0.4704, 0.207971, 0.025791,
    0.403993, 0.251412, -0.01831, 0.277838,
    -0.11047, 0.066928, 0.128539, -0.18911,
    0.133558, -0.02105, 149.62
]

# --------------------------------------------------
# Prediction
# --------------------------------------------------
response = endpoint.predict(instances=[instance])

prediction = response.predictions[0]

print("✅ Raw prediction:", prediction)

# Optional: Interpret result
if int(prediction) == 1:
    print("🚨 Fraud detected")
else:
    print("✅ Transaction is legitimate")
