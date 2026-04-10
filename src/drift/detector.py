from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingestion import FEATURE_COLUMNS
from src.utils.settings import get_settings

logger = logging.getLogger(__name__)


def _psi_for_column(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    epsilon = 1e-8
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(reference, quantiles))

    if len(breakpoints) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_perc = ref_counts / max(ref_counts.sum(), 1)
    cur_perc = cur_counts / max(cur_counts.sum(), 1)

    ref_perc = np.clip(ref_perc, epsilon, None)
    cur_perc = np.clip(cur_perc, epsilon, None)

    psi = np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc))
    return float(psi)


def detect_data_drift(
    reference_path: str | Path | None = None,
    incoming_path: str | Path | None = None,
    threshold: float | None = None,
) -> dict:
    settings = get_settings()

    reference_path = Path(reference_path) if reference_path else settings.processed_data_dir / "reference_features.csv"
    incoming_path = Path(incoming_path) if incoming_path else settings.raw_data_dir / "incoming.csv"

    reference_df = pd.read_csv(reference_path)
    incoming_df = pd.read_csv(incoming_path)

    drift_scores: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        drift_scores[col] = _psi_for_column(reference_df[col], incoming_df[col])

    overall_score = float(np.mean(list(drift_scores.values())))
    threshold = threshold if threshold is not None else settings.drift_threshold
    drift_detected = overall_score >= threshold

    result = {
        "drift_detected": drift_detected,
        "drift_score": overall_score,
        "threshold": threshold,
        "feature_scores": drift_scores,
    }

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.models_dir / "drift_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info("Drift detection complete. score=%.4f threshold=%.4f", overall_score, threshold)
    return result


if __name__ == "__main__":
    from src.utils.logging_utils import configure_logging

    configure_logging()
    logger.info("Drift result: %s", detect_data_drift())
