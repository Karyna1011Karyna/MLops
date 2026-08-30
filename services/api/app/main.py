#TarotRAG Intent Classifier API

import asyncio
import logging
import os
import random
import time

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from common import db
from training.train_baseline import (
    load_dataset_from_minio,
    load_local_seed,
    log_run,
    preprocess_dataset,
    register_and_evaluate,
    train,
)

from .schemas import FeedbackRequest, FeedbackResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tarotrag.api")

app = FastAPI(
    title="TarotRAG — Intent Classifier API",
    description="Baseline ML component of the TarotRAG design doc: classifies a "
                 "tarot question into a domain (love/career/health/...) so the RAG "
                 "step downstream can narrow its retrieval. Served via an MLflow "
                 "champion/challenger scheme.",
    version="0.2.0",
)

MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "tarot-intent-classifier")
CHALLENGER_TRAFFIC_PCT = float(os.environ.get("CHALLENGER_TRAFFIC_PCT", "10"))
MODEL_REFRESH_SECONDS = float(os.environ.get("MODEL_REFRESH_SECONDS", "60"))

MODEL_STATE = {
    "champion": None, "champion_version": None,
    "challenger": None, "challenger_version": None,
}


def _load_alias(alias: str):
    #Returns (pipeline, version) for a registry alias or (None, None)
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, alias)
        pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
        return pipeline, mv.version
    except MlflowException:
        return None, None


def _bootstrap_if_empty():
    #First-ever run of the whole system
    logger.warning("No champion registered yet — training a bootstrap baseline.")
    df, dataset_version = load_dataset_from_minio()
    if df is None:
        df, dataset_version = load_local_seed(), "local-seed"
    df = preprocess_dataset(df)
    pipeline, metrics, report = train(df)
    run_id = log_run(pipeline, metrics, report, dataset_version, df, run_name=f"bootstrap-{dataset_version}")
    version, role, _promoted = register_and_evaluate(run_id, metrics)
    logger.info("Bootstrap model registered as version %s -> %s", version, role)


def _refresh_models():
    champion, champion_version = _load_alias("champion")
    if champion is None:
        _bootstrap_if_empty()
        champion, champion_version = _load_alias("champion")
    challenger, challenger_version = _load_alias("challenger")

    changed = (
        champion_version != MODEL_STATE["champion_version"]
        or challenger_version != MODEL_STATE["challenger_version"]
    )
    MODEL_STATE["champion"], MODEL_STATE["champion_version"] = champion, champion_version
    MODEL_STATE["challenger"], MODEL_STATE["challenger_version"] = challenger, challenger_version
    if changed:
        logger.info("Model state refreshed: champion=%s challenger=%s", champion_version, challenger_version)


async def _refresh_loop():
    while True:
        await asyncio.sleep(MODEL_REFRESH_SECONDS)
        try:
            await asyncio.to_thread(_refresh_models)
        except Exception as exc:  # noqa: BLE001 - a failed background refresh must not crash the process
            logger.warning("Background model refresh failed: %s", exc)


@app.on_event("startup")
async def startup():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    last_error = None
    for attempt in range(1, 11):
        try:
            await asyncio.to_thread(_refresh_models)
            break
        except Exception as exc:
            last_error = exc
            logger.warning("Model bootstrap attempt %s/10 failed: %s", attempt, exc)
            time.sleep(3)
    else:
        raise RuntimeError("Could not load or train a model after 10 attempts — "
                            "is MLflow/MinIO reachable?") from last_error

    asyncio.create_task(_refresh_loop())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "champion_version": MODEL_STATE["champion_version"],
        "challenger_version": MODEL_STATE["challenger_version"],
        "challenger_traffic_pct": CHALLENGER_TRAFFIC_PCT,
    }


@app.post("/admin/reload-models")
def reload_models():
    #Force-refresh champion/challenger from the registry right now,

    return {
        "champion_version": MODEL_STATE["champion_version"],
        "challenger_version": MODEL_STATE["challenger_version"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if MODEL_STATE["champion"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    use_challenger = (
        MODEL_STATE["challenger"] is not None
        and random.random() < CHALLENGER_TRAFFIC_PCT / 100
    )
    served_by = "challenger" if use_challenger else "champion"
    pipeline = MODEL_STATE[served_by]
    version = MODEL_STATE[f"{served_by}_version"]

    probabilities = pipeline.predict_proba([payload.question])[0]
    proba_map = {label: float(p) for label, p in zip(pipeline.classes_, probabilities)}
    best_label = max(proba_map, key=proba_map.get)

    return PredictResponse(
        intent=best_label,
        confidence=proba_map[best_label],
        probabilities=proba_map,
        served_by=served_by,
        model_version=version,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest):
    #Record a confirmed (question, label) pair as new training data
    example_id = db.insert_training_example(payload.question, payload.label, source="api_feedback")
    return FeedbackResponse(id=example_id, status="stored")
