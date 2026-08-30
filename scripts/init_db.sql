CREATE TABLE IF NOT EXISTS training_examples (
    id                 SERIAL PRIMARY KEY,
    question           TEXT NOT NULL,
    label              TEXT NOT NULL,
    source             TEXT NOT NULL DEFAULT 'unknown',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    synced_to_dataset  BOOLEAN NOT NULL DEFAULT false,
    synced_at          TIMESTAMPTZ,
    dataset_version    TEXT
);

--  Airflow DAG:  give me everything not yet synced query
CREATE INDEX IF NOT EXISTS idx_training_examples_unsynced
    ON training_examples (synced_to_dataset)
    WHERE synced_to_dataset = false;
