import joblib
import numpy as np
import os
import sys

class Predictor:
    def __init__(self):
        print("🔥🔥🔥 CUSTOM PREDICTOR INIT 🔥🔥🔥", file=sys.stderr)

        model_dir = os.environ.get("AIP_MODEL_DIR")
        if not model_dir:
            raise RuntimeError("AIP_MODEL_DIR not set")

        model_path = os.path.join(model_dir, "model.pkl")
        print(f"📦 Loading model from: {model_path}", file=sys.stderr)

        self.model = joblib.load(model_path)

    def predict(self, instances):
        print("🔥🔥🔥 CUSTOM PREDICT CALLED 🔥🔥🔥", file=sys.stderr)

        X = np.array(instances)
        preds = self.model.predict(X)
        return preds.tolist()
