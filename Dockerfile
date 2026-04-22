FROM us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest

WORKDIR /app

COPY predictor.py /app/predictor.py

ENV AIP_PREDICTOR_CLASS=predictor.Predictor