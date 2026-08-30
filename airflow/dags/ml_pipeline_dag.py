#Full ML pipeline: gather -> process -> tune -> train -> register -> evaluate/promote.


from __future__ import annotations

from datetime import datetime

import pandas as pd
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from common import storage
from training.train_baseline import (
    configure_mlflow,
    load_dataset_from_minio,
    log_run,
    preprocess_dataset,
    register_and_evaluate,
    train,
    tune_hyperparameters,
)

DEFAULT_ARGS = {"owner": "tarotrag", "retries": 1}

with DAG(
    dag_id="ml_pipeline",
    description="Full pipeline: gather data (sync_training_data) -> process -> "
                "tune -> train -> register -> evaluate -> promote if better.",
    default_args=DEFAULT_ARGS,
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tarotrag", "intent-classifier", "pipeline"],
) as dag:

    # data gathering

    gather_data = TriggerDagRunOperator(
        task_id="gather_data",
        trigger_dag_id="sync_training_data",
        wait_for_completion=True,
        poke_interval=5,
        deferrable=True,
    )

    @task
    def process_data() -> dict:
        df, dataset_version = load_dataset_from_minio()
        if df is None:
            raise ValueError("No dataset in MinIO yet — run scripts/publish_seed_dataset.py first.")

        before = len(df)
        df = preprocess_dataset(df)
        if len(df) != before:
            # rows were dropped -> the cleaned set is a new, distinct version
            dataset_version = storage.write_dataset_version(df.to_dict("records"))

        return {"dataset_version": dataset_version, "n_rows": len(df)}

    @task
    def tune_and_train(processed: dict) -> dict:
        rows, dataset_version = storage.read_dataset(version=processed["dataset_version"])
        df = pd.DataFrame(rows)

        tuning = tune_hyperparameters(df)
        pipeline, metrics, report = train(df, tuning["vectorizer_params"], tuning["classifier_params"])
        metrics["cv_best_macro_f1"] = float(tuning["cv_best_macro_f1"])

        configure_mlflow()
        run_id = log_run(
            pipeline, metrics, report, dataset_version, df,
            run_name=f"pipeline-{dataset_version}",
            vectorizer_params=tuning["vectorizer_params"],
            classifier_params=tuning["classifier_params"],
            extra_params={"tuned": True, "triggered_by": "airflow_ml_pipeline"},
        )
        numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        return {"run_id": run_id, "metrics": numeric_metrics}

    @task
    def evaluate_and_promote(trained: dict) -> dict:
        configure_mlflow()
        version, role, promoted = register_and_evaluate(trained["run_id"], trained["metrics"])
        return {"version": version, "role": role, "promoted": promoted}

    processed_result = process_data()
    gather_data >> processed_result

    trained_result = tune_and_train(processed_result)
    evaluate_and_promote(trained_result)
