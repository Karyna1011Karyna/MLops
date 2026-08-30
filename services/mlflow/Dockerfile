# Our own MLflow tracking server + model registry, self-hosted.
#
# Backend store (experiments, runs, params, metrics, registered models,
# aliases) lives in the `mlflow` database on the same Postgres instance the
# rest of the stack uses. Artifact store (the actual model files) lives in
# the `mlflow-artifacts` bucket on the same MinIO instance — see the
# MLFLOW_S3_ENDPOINT_URL / AWS_* env vars wired up in docker-compose.yml.
FROM python:3.11-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
