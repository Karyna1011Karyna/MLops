#One-off: publish the local seed CSV as version v0-seed of the intent dataset in MinIO
import csv
import os

from common import storage

SEED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "seed_intent_dataset.csv")


def main():
    if storage.latest_dataset_manifest() is not None:
        print("A dataset already exists in MinIO — skipping seed publish.")
        return

    with open(SEED_PATH, newline="", encoding="utf-8") as f:
        rows = [{"question": r["question"], "label": r["label"]} for r in csv.DictReader(f)]

    version = storage.write_dataset_version(rows, version="v0-seed")
    print(f"Published seed dataset as version '{version}' ({len(rows)} rows).")


if __name__ == "__main__":
    main()
