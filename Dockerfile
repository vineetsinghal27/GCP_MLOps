FROM us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest

WORKDIR /app

COPY predictor.py /app/predictor.py

# ✅ THIS LINE FIXES THE ERROR
ENV PYTHONPATH=/app:/usr/lib/python3.10:/opt/conda/lib/python3.10/site-packages

ENV AIP_PREDICTOR_CLASS=predictor.Predictor
