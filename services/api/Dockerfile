# Build context is the repo root (tarotrag/), not this directory —
# see docker-compose.yml's `build.context: .` for this service.
FROM python:3.11-slim

WORKDIR /srv
ENV PYTHONPATH=/srv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY services/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY training ./training
COPY data ./data
COPY scripts ./scripts
COPY services/api/app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
