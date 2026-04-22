FROM us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest

COPY predictor.py /predictor.py

ENV AIP_PREDICTOR_CLASS=predictor.Predictor
