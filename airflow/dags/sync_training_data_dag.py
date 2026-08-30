# move newly-labeled questions from Postgres into a new versioned training-dataset file in MinIO.

from __future__ import annotations

from datetime import datetime
from typing import List

from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.operators.python import ShortCircuitOperator

from common import db, storage

DEFAULT_ARGS = {
    "owner": "tarotrag",
    "retries": 1,
}


def _has_new_rows(rows: List[dict]) -> bool:
    return bool(rows)


with DAG(
    dag_id="sync_training_data",
    description="Move new labeled questions from Postgres into a versioned "
                "training-dataset file in MinIO for the intent classifier.",
    default_args=DEFAULT_ARGS,
    schedule=None,  # manual trigger for the demo
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tarotrag", "intent-classifier"],
) as dag:

    @task
    def extract_new_rows() -> List[dict]:
        return db.fetch_unsynced_examples()

    @task
    def append_to_dataset(new_rows: List[dict]) -> dict:
        existing_rows, _ = storage.read_dataset()
        existing_rows = existing_rows or []
        combined = existing_rows + [{"question": r["question"], "label": r["label"]} for r in new_rows]
        version = storage.write_dataset_version(combined)
        return {"version": version, "ids": [r["id"] for r in new_rows]}

    @task
    def mark_synced(sync_result: dict) -> None:
        db.mark_examples_synced(sync_result["ids"], sync_result["version"])

    new_rows = extract_new_rows()
    gate = ShortCircuitOperator(
        task_id="skip_if_no_new_data",
        python_callable=_has_new_rows,
        op_kwargs={"rows": new_rows},
    )
    sync_result = append_to_dataset(new_rows)

    new_rows >> gate >> sync_result >> mark_synced(sync_result)
