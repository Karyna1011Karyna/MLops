#MinIO / S3 helpers shared by the API, the training script and the Airflow DAG

import json
import os
from datetime import datetime, timezone
from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

DATASET_PREFIX = "intent"


def _client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ROOT_USER"],
        aws_secret_access_key=os.environ["MINIO_ROOT_PASSWORD"],
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _datasets_bucket() -> str:
    return os.environ.get("MINIO_DATASETS_BUCKET", "tarot-datasets")


def new_version_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json_key(bucket: str, key: str) -> Optional[dict]:
    try:
        obj = _client().get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NoSuchBucket"):
            return None
        raise


def _write_json_key(bucket: str, key: str, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _client().put_object(Bucket=bucket, Key=key, Body=body)


# --- dataset ---

def latest_dataset_manifest() -> Optional[dict]:
    return _read_json_key(_datasets_bucket(), f"{DATASET_PREFIX}/latest.json")


def read_dataset(version: Optional[str] = None):
    #Return (rows, version) where rows is list[{"question", "label"}].
    #Returns (None, None) if no dataset has been published yet.

    if version is None:
        manifest = latest_dataset_manifest()
        if manifest is None:
            return None, None
        key, version = manifest["key"], manifest["version"]
    else:
        key = f"{DATASET_PREFIX}/dataset_{version}.jsonl"

    obj = _client().get_object(Bucket=_datasets_bucket(), Key=key)
    text = obj["Body"].read().decode("utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows, version


def write_dataset_version(rows: list, version: Optional[str] = None) -> str:
    # Upload row as a new immutable dataset version and repoint latest.json at it
    version = version or new_version_tag()
    key = f"{DATASET_PREFIX}/dataset_{version}.jsonl"
    body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows).encode("utf-8")
    _client().put_object(Bucket=_datasets_bucket(), Key=key, Body=body)
    _write_json_key(
        _datasets_bucket(),
        f"{DATASET_PREFIX}/latest.json",
        {"version": version, "key": key, "row_count": len(rows)},
    )
    return version
