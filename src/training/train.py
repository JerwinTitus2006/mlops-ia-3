from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.data.ingestion import TARGET_COLUMN
from src.training.model_registry import register_model_version
from src.utils.settings import get_settings

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


def train_model(
    run_name: str = "training",
    n_estimators: int = 200,
    max_depth: int = 8,
    min_samples_split: int = 2,
) -> dict:
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(settings.processed_data_dir / "X_train.csv")
    X_test = pd.read_csv(settings.processed_data_dir / "X_test.csv")
    y_train = pd.read_csv(settings.processed_data_dir / "y_train.csv")[TARGET_COLUMN]
    y_test = pd.read_csv(settings.processed_data_dir / "y_test.csv")[TARGET_COLUMN]

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(settings.tracking_uri)
        mlflow.set_experiment("self_healing_ml_pipeline")
    else:
        logger.warning("MLflow is not available in this Python environment. Skipping experiment tracking.")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=settings.random_state,
    )

    run_context = mlflow.start_run(run_name=run_name) if MLFLOW_AVAILABLE else None
    with run_context if run_context else _NoOpContext() as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = float(accuracy_score(y_test, predictions))
        f1 = float(f1_score(y_test, predictions, average="weighted"))

        if MLFLOW_AVAILABLE:
            mlflow.log_params(
                {
                    "model_type": "RandomForestClassifier",
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "random_state": settings.random_state,
                }
            )
            mlflow.log_metrics({"accuracy": accuracy, "f1_weighted": f1})
            mlflow.sklearn.log_model(model, artifact_path="model")

        temp_model_path = settings.models_dir / "_latest_candidate.joblib"
        joblib.dump(model, temp_model_path)
        run_id = run.info.run_id if run else "local_no_mlflow"
        model_record = register_model_version(temp_model_path, accuracy=accuracy, run_id=run_id)
        temp_model_path.unlink(missing_ok=True)

        metadata = {
            "run_id": run_id,
            "accuracy": accuracy,
            "f1_weighted": f1,
            "current_version": model_record["version"],
        }

        metadata_path = settings.models_dir / "current_model_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model trained. Accuracy=%.4f, version=v%s", accuracy, model_record["version"])
        return metadata


class _NoOpContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    from src.utils.logging_utils import configure_logging

    configure_logging()
    output = train_model(run_name="manual_train")
    logger.info("Training output: %s", output)
