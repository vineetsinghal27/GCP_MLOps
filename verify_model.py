import joblib

model = joblib.load("outputs/model.pkl")
print("✅ Loaded model type:", type(model))

if not hasattr(model, "predict"):
    raise ValueError("❌ Loaded object is not a sklearn model (no predict method)")
