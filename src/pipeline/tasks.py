from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data.ingestion import ingest_data
from src.drift.detector import detect_data_drift
from src.drift.performance_monitor import evaluate_live_performance
from src.features.preprocessing import preprocess_data
from src.training.self_heal import run_self_healing_cycle, should_trigger_retraining
from src.training.train import train_model
from src.utils.settings import get_settings

logger = logging.getLogger(__name__)


def _write_runtime_report(name: str, payload: dict) -> str:
    settings = get_settings()
    runtime_dir = settings.models_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def task_ingest_data() -> dict:
    result = ingest_data(simulate_drift=True)
    _write_runtime_report("ingestion", result)
    return result


def task_preprocess_data() -> dict:
    result = preprocess_data()
    _write_runtime_report("preprocessing", result)
    return result


def task_train_baseline() -> dict:
    settings = get_settings()
    current_model_path = settings.models_dir / "current_model.joblib"
    if current_model_path.exists():
        return {"action": "skipped", "reason": "baseline model already exists"}

    result = train_model(run_name="baseline_train")
    _write_runtime_report("baseline_training", result)
    return result


def task_detect_drift() -> dict:
    result = detect_data_drift()
    _write_runtime_report("drift_detection", result)
    return result


def task_check_performance() -> dict:
    result = evaluate_live_performance()
    _write_runtime_report("performance", result)
    return result


def task_retrain_decision() -> dict:
    result = should_trigger_retraining()
    _write_runtime_report("retrain_decision", result)
    return result


def task_retrain_if_needed() -> dict:
    result = run_self_healing_cycle()
    _write_runtime_report("self_healing", result)
    return result
