# PostgreSQL helpers for the training_examples table
import os
from typing import Iterable, List

import psycopg2
import psycopg2.extras


def get_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )


def insert_training_example(question: str, label: str, source: str = "api_feedback") -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO training_examples (question, label, source) "
                "VALUES (%s, %s, %s) RETURNING id",
                (question, label, source),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
    return new_id


def fetch_unsynced_examples() -> List[dict]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, question, label, source FROM training_examples "
                "WHERE synced_to_dataset = false ORDER BY id"
            )
            return [dict(row) for row in cur.fetchall()]


def mark_examples_synced(ids: Iterable[int], dataset_version: str) -> None:
    ids = list(ids)
    if not ids:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE training_examples SET synced_to_dataset = true, synced_at = now(), "
                "dataset_version = %s WHERE id = ANY(%s)",
                (dataset_version, ids),
            )
        conn.commit()
