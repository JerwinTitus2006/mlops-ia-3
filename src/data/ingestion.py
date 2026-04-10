from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.utils.settings import get_settings

logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "target"


def _apply_synthetic_shift(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shifted = df.copy()

    # Simulate realistic drift in feature distributions.
    shifted["sepal_length"] = shifted["sepal_length"] + rng.normal(0.4, 0.2, len(shifted))
    shifted["petal_width"] = shifted["petal_width"] * rng.normal(1.15, 0.05, len(shifted))
    shifted["sepal_width"] = shifted["sepal_width"] + rng.normal(-0.2, 0.1, len(shifted))
    return shifted


def ingest_data(simulate_drift: bool = True) -> dict:
    settings = get_settings()
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": TARGET_COLUMN,
        }
    )

    baseline, incoming = train_test_split(
        df,
        test_size=0.2,
        random_state=settings.random_state,
        stratify=df[TARGET_COLUMN],
    )

    if simulate_drift:
        incoming = _apply_synthetic_shift(incoming, seed=settings.random_state)

    baseline_path = settings.raw_data_dir / "baseline.csv"
    incoming_path = settings.raw_data_dir / "incoming.csv"

    baseline.to_csv(baseline_path, index=False)
    incoming.to_csv(incoming_path, index=False)

    logger.info("Raw baseline data saved to %s", baseline_path)
    logger.info("Raw incoming data saved to %s", incoming_path)

    return {
        "baseline_path": str(baseline_path),
        "incoming_path": str(incoming_path),
        "rows_baseline": len(baseline),
        "rows_incoming": len(incoming),
    }


if __name__ == "__main__":
    from src.utils.logging_utils import configure_logging

    configure_logging()
    result = ingest_data(simulate_drift=True)
    logger.info("Ingestion result: %s", result)
