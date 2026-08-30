#Champion/challenger promotion: make the current challenger the new

#docker compose run --rm promote-challenger

import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "tarot-intent-classifier")


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    try:
        challenger = client.get_model_version_by_alias(MODEL_NAME, "challenger")
    except MlflowException:
        print("No challenger is currently registered — nothing to promote. "
              "Run training/train_baseline.py --register first.")
        return

    try:
        old_champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        old_version = old_champion.version
    except MlflowException:
        old_version = None

    client.set_registered_model_alias(MODEL_NAME, "champion", challenger.version)
    client.delete_registered_model_alias(MODEL_NAME, "challenger")

    print(f"Promoted {MODEL_NAME} version {challenger.version} to champion "
          f"(was version {old_version}).")
    print("The old champion version is still in the registry, just unaliased — "
          "nothing was deleted, so this is safe to reverse by hand if needed.")


if __name__ == "__main__":
    main()
