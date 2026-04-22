import joblib
import numpy as np

class Predictor:
    def __init__(self):
        # Vertex AI mounts the model here automatically
        self.model = joblib.load("/aiplatform/model/model.pkl")

    def predict(self, instances):
        X = np.array(instances)
        predictions = self.model.predict(X)
        return predictions.tolist()
