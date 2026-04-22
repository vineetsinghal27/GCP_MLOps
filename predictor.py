import joblib
import numpy as np
import sys

class Predictor:
    def __init__(self):
        print("🔥🔥🔥 CUSTOM PREDICTOR INIT 🔥🔥🔥", file=sys.stderr)
        self.model = joblib.load("/aiplatform/model/model.pkl")

    def predict(self, instances):
        print("🔥🔥🔥 CUSTOM PREDICT CALLED 🔥🔥🔥", file=sys.stderr)
        X = np.array(instances)
        return self.model.predict(X).tolist()