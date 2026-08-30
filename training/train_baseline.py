"""Train the TarotRAG intent-classifier baseline: TF-IDF + Logistic Regression.

This is the "main ML component" from the design doc — a small, fast,
fully-interpretable baseline. It is intentionally NOT a transformer: a
baseline should be the simplest thing that could possibly work, so later
improvements (e.g. a fine-tuned DistilBERT) have something honest to beat.

Full pipeline, step by step (also wired up as the Airflow DAG `ml_pipeline`):
    1. load_dataset_from_minio / load_local_seed  — data gathering
    2. preprocess_dataset                          — data processing
    3. tune_hyperparameters (optional, --tune)      — parameter tuning
    4. train                                        — model training
    5. log_run                                      — always-on MLflow tracking
    6. register_and_evaluate (--register)           — registry + auto champion/challenger

Every run is tracked in MLflow (params, metrics, the classification report,
the model artifact itself — stored in MinIO via MLflow's S3 artifact
store). Only runs started with --register also get registered as a new
version of the `tarot-intent-classifier` model and evaluated against the
current champion:

    - no "champion" alias exists yet        -> this version BECOMES the champion
      (first-ever model, nothing to compare it against)
    - a champion exists, new macro_f1 higher -> AUTOMATICALLY promoted to champion
    - a champion exists, new macro_f1 lower  -> stays a "challenger", visible in the
      registry for inspection, but does not receive production traffic

Usage:
    python training/train_baseline.py --source minio --tune --register
    python training/train_baseline.py --source local              # just track, don't deploy
"""
import argparse
import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from common import storage

LOCAL_SEED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "seed_intent_dataset.csv")
EXPERIMENT_NAME = "tarot-intent-classifier"
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "tarot-intent-classifier")
PROMOTION_METRIC = "macro_f1"

# Defaults used when --tune is NOT passed — picked by hand via an earlier
# 5-fold CV sweep (see git history / the design doc for that experiment).
VECTORIZER_PARAMS = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
CLASSIFIER_PARAMS = dict(max_iter=2000, class_weight="balanced", C=3)

# Search space for --tune: small on purpose — this dataset is a few hundred
# rows, so an exhaustive grid over a handful of options is instant and
# already covers the range worth trying by hand.
PARAM_GRID = {
    "tfidf__ngram_range": [(2, 3), (2, 4), (2, 5), (3, 5)],
    "clf__C": [1, 3, 5, 10],
}


def load_dataset_from_minio():
    """Returns (DataFrame, version) or (None, None) if nothing has been published yet."""
    rows, version = storage.read_dataset()
    if rows is None:
        return None, None
    return pd.DataFrame(rows), version


def load_local_seed() -> pd.DataFrame:
    return pd.read_csv(LOCAL_SEED_PATH)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Data-processing step: drop empty/malformed rows and exact duplicates
    (the dataset only ever grows via appends from Postgres, so duplicates
    from a re-run or a repeated piece of feedback are expected)."""
    before = len(df)

    df = df.dropna(subset=["question", "label"]).copy()
    df["question"] = df["question"].str.strip()
    df["label"] = df["label"].str.strip()
    df = df[df["question"].str.len() > 0]

    df = df.drop_duplicates(subset=["question", "label"]).reset_index(drop=True)

    removed = before - len(df)
    if removed:
        print(f"Data processing: removed {removed} empty/duplicate row(s) ({before} -> {len(df)}).")
    return df


def build_pipeline(vectorizer_params: dict = None, classifier_params: dict = None) -> Pipeline:
    # Character n-grams (rather than word n-grams) handle Ukrainian's rich
    # inflection much better on a small dataset: "стосунків"/"стосунки"/
    # "стосунках" share substrings a word-level vectorizer would treat as
    # unrelated tokens.
    return Pipeline([
        ("tfidf", TfidfVectorizer(**(vectorizer_params or VECTORIZER_PARAMS))),
        ("clf", LogisticRegression(**(classifier_params or CLASSIFIER_PARAMS))),
    ])


def tune_hyperparameters(df: pd.DataFrame) -> dict:
    """Parameter-tuning step: small grid search over vectorizer n-gram range
    and classifier regularization strength, scored by 5-fold CV macro-F1.
    Returns the best vectorizer/classifier params plus the CV score, ready
    to hand straight to train()."""
    base_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(base_pipeline, PARAM_GRID, scoring="f1_macro", cv=cv, refit=False)
    search.fit(df["question"], df["label"])

    best = search.best_params_
    vectorizer_params = dict(VECTORIZER_PARAMS, ngram_range=best["tfidf__ngram_range"])
    classifier_params = dict(CLASSIFIER_PARAMS, C=best["clf__C"])

    print(f"Grid search: best cv macro_f1={search.best_score_:.3f} "
          f"with ngram_range={best['tfidf__ngram_range']}, C={best['clf__C']} "
          f"(searched {len(search.cv_results_['params'])} combinations x 5 folds).")

    return {
        "vectorizer_params": vectorizer_params,
        "classifier_params": classifier_params,
        "cv_best_macro_f1": search.best_score_,
    }


def train(df: pd.DataFrame, vectorizer_params: dict = None, classifier_params: dict = None):
    x_train, x_test, y_train, y_test = train_test_split(
        df["question"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    pipeline = build_pipeline(vectorizer_params, classifier_params)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "classes": sorted(df["label"].unique().tolist()),
    }
    report = classification_report(y_test, y_pred, zero_division=0)
    return pipeline, metrics, report


def configure_mlflow():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)


def _has_alias(client: MlflowClient, alias: str) -> bool:
    try:
        client.get_model_version_by_alias(MODEL_NAME, alias)
        return True
    except MlflowException:
        return False


def _champion_metric(client: MlflowClient, metric_name: str):
    """Returns (champion_version, champion_metric_value) or (None, None) if
    no champion is registered yet."""
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None, None
    run = client.get_run(champion.run_id)
    return champion.version, run.data.metrics.get(metric_name)


def log_run(pipeline: Pipeline, metrics: dict, report: str, dataset_version: str,
            df: pd.DataFrame, run_name: str, vectorizer_params: dict = None,
            classifier_params: dict = None, extra_params: dict = None) -> str:
    """Always-on MLflow tracking: params, metrics, the report, and the model
    artifact (uploaded to MinIO under the hood). Returns the MLflow run_id."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("n_examples", len(df))
        mlflow.log_params({f"vectorizer_{k}": v for k, v in (vectorizer_params or VECTORIZER_PARAMS).items()})
        mlflow.log_params({f"clf_{k}": v for k, v in (classifier_params or CLASSIFIER_PARAMS).items()})
        if extra_params:
            mlflow.log_params(extra_params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_text(report, "classification_report.txt")
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        return run.info.run_id


def register_and_evaluate(run_id: str, new_metrics: dict, metric_name: str = PROMOTION_METRIC) -> tuple:
    """Register the model logged in `run_id` as a new version of MODEL_NAME,
    then decide its role automatically by comparing `metric_name` against
    the current champion's own logged value for that same metric — this is
    the pipeline's evaluation gate, not a human eyeballing the MLflow UI.

    Returns (version, role_description, was_promoted).
    """
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri, MODEL_NAME)

    champion_version, champion_score = _champion_metric(client, metric_name)
    new_score = new_metrics.get(metric_name)

    if champion_version is None:
        client.set_registered_model_alias(MODEL_NAME, "champion", model_version.version)
        return model_version.version, "champion (bootstrap — no previous champion existed)", True

    client.set_registered_model_alias(MODEL_NAME, "challenger", model_version.version)

    is_better = new_score is not None and champion_score is not None and new_score > champion_score
    if is_better:
        client.set_registered_model_alias(MODEL_NAME, "champion", model_version.version)
        client.delete_registered_model_alias(MODEL_NAME, "challenger")
        role = (f"promoted to champion automatically "
                f"({metric_name}={new_score:.3f} > previous champion's {champion_score:.3f}, "
                f"was version {champion_version})")
        return model_version.version, role, True

    role = (f"kept as challenger, NOT promoted "
            f"({metric_name}={new_score:.3f} <= champion v{champion_version}'s {champion_score:.3f})")
    return model_version.version, role, False


def main():
    parser = argparse.ArgumentParser(description="Train the TarotRAG intent-classifier baseline.")
    parser.add_argument("--source", choices=["minio", "local"], default="minio",
                         help="Where to read the training dataset from (default: minio).")
    parser.add_argument("--tune", action="store_true",
                         help="Run a small grid search over vectorizer/classifier hyperparameters "
                              "before the final training run, instead of using the hardcoded defaults.")
    parser.add_argument("--register", action="store_true",
                         help="Register this run's model as a new version and evaluate it against "
                              "the current champion (macro_f1), auto-promoting if it wins. Without "
                              "this flag the run is only tracked in MLflow, not served.")
    args = parser.parse_args()

    if args.source == "minio":
        df, dataset_version = load_dataset_from_minio()
        if df is None:
            print("No dataset found in MinIO yet — falling back to the local seed CSV.")
            df, dataset_version = load_local_seed(), "local-seed"
    else:
        df, dataset_version = load_local_seed(), "local-seed"

    df = preprocess_dataset(df)
    print(f"Training on {len(df)} examples across {df['label'].nunique()} classes "
          f"(dataset version: {dataset_version})")

    vectorizer_params, classifier_params = None, None
    extra_params = {"tuned": args.tune}
    if args.tune:
        tuning = tune_hyperparameters(df)
        vectorizer_params = tuning["vectorizer_params"]
        classifier_params = tuning["classifier_params"]

    pipeline, metrics, report = train(df, vectorizer_params, classifier_params)
    if args.tune:
        metrics["cv_best_macro_f1"] = float(tuning["cv_best_macro_f1"])
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(report)

    configure_mlflow()
    run_id = log_run(pipeline, metrics, report, dataset_version, df, run_name=f"baseline-{dataset_version}",
                      vectorizer_params=vectorizer_params, classifier_params=classifier_params,
                      extra_params=extra_params)
    print(f"Tracked as MLflow run {run_id}.")

    if args.register:
        version, role, promoted = register_and_evaluate(run_id, metrics)
        print(f"Registered as {MODEL_NAME} version {version} -> {role}")
    else:
        print("Not registered (pass --register to make this run deployable).")


if __name__ == "__main__":
    main()
