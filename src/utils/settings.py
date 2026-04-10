from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    models_dir: Path
    mlruns_dir: Path
    model_name: str
    tracking_uri: str
    drift_threshold: float
    performance_threshold: float
    min_acceptable_improvement: float
    random_state: int


def get_settings() -> Settings:
    project_root = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    data_dir = project_root / "data"

    mlruns_dir = project_root / "mlruns"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file:{mlruns_dir}")

    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        raw_data_dir=data_dir / "raw",
        processed_data_dir=data_dir / "processed",
        models_dir=project_root / "models",
        mlruns_dir=mlruns_dir,
        model_name=os.getenv("MODEL_NAME", "iris_classifier"),
        tracking_uri=tracking_uri,
        drift_threshold=float(os.getenv("DRIFT_THRESHOLD", "0.2")),
        performance_threshold=float(os.getenv("PERFORMANCE_THRESHOLD", "0.85")),
        min_acceptable_improvement=float(os.getenv("MIN_ACCEPTABLE_IMPROVEMENT", "0.0")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
    )
