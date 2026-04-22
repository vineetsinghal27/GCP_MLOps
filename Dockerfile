FROM us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest

# ✅ Copy into /usr/app (expected by Vertex images)
COPY predictor.py /usr/app/predictor.py

# ✅ Tell Vertex which class to load
ENV AIP_PREDICTOR_CLASS=predictor.Predictor
