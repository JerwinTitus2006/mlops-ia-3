from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

from src.data.ingestion import FEATURE_COLUMNS, TARGET_COLUMN
from src.utils.settings import get_settings

logger = logging.getLogger(__name__)


def evaluate_live_performance(incoming_path: str | Path | None = None) -> dict:
    settings = get_settings()
    incoming_path = Path(incoming_path) if incoming_path else settings.raw_data_dir / "incoming.csv"
    model_path = settings.models_dir / "current_model.joblib"

    if not model_path.exists():
        return {
            "has_model": False,
            "accuracy": None,
            "threshold": settings.performance_threshold,
            "performance_drop": False,
        }

    incoming = pd.read_csv(incoming_path)
    if TARGET_COLUMN not in incoming.columns:
        return {
            "has_model": True,
            "accuracy": None,
            "threshold": settings.performance_threshold,
            "performance_drop": False,
            "note": "Incoming data has no target column",
        }

    model = joblib.load(model_path)
    predictions = model.predict(incoming[FEATURE_COLUMNS])
    accuracy = float(accuracy_score(incoming[TARGET_COLUMN], predictions))
    performance_drop = accuracy < settings.performance_threshold

    logger.info("Live performance accuracy=%.4f threshold=%.4f", accuracy, settings.performance_threshold)

    return {
        "has_model": True,
        "accuracy": accuracy,
        "threshold": settings.performance_threshold,
        "performance_drop": performance_drop,
    }
