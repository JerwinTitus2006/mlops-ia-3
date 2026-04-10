from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.data.ingestion import FEATURE_COLUMNS, TARGET_COLUMN
from src.utils.settings import get_settings

logger = logging.getLogger(__name__)


def preprocess_data(raw_baseline_path: str | Path | None = None) -> dict:
    settings = get_settings()
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(raw_baseline_path) if raw_baseline_path else settings.raw_data_dir / "baseline.csv"
    df = pd.read_csv(baseline_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=settings.random_state,
        stratify=y,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURE_COLUMNS)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=FEATURE_COLUMNS)

    X_train_path = settings.processed_data_dir / "X_train.csv"
    X_test_path = settings.processed_data_dir / "X_test.csv"
    y_train_path = settings.processed_data_dir / "y_train.csv"
    y_test_path = settings.processed_data_dir / "y_test.csv"
    reference_path = settings.processed_data_dir / "reference_features.csv"
    imputer_path = settings.models_dir / "imputer.joblib"
    feature_config_path = settings.processed_data_dir / "feature_columns.json"

    settings.models_dir.mkdir(parents=True, exist_ok=True)

    X_train_imputed.to_csv(X_train_path, index=False)
    X_test_imputed.to_csv(X_test_path, index=False)
    y_train.to_frame(TARGET_COLUMN).to_csv(y_train_path, index=False)
    y_test.to_frame(TARGET_COLUMN).to_csv(y_test_path, index=False)
    X_train_imputed.to_csv(reference_path, index=False)

    import joblib

    joblib.dump(imputer, imputer_path)
    with feature_config_path.open("w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    logger.info("Preprocessed datasets saved in %s", settings.processed_data_dir)

    return {
        "X_train_path": str(X_train_path),
        "X_test_path": str(X_test_path),
        "y_train_path": str(y_train_path),
        "y_test_path": str(y_test_path),
        "reference_path": str(reference_path),
        "imputer_path": str(imputer_path),
    }


if __name__ == "__main__":
    from src.utils.logging_utils import configure_logging

    configure_logging()
    logger.info("Preprocess result: %s", preprocess_data())
