from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.training.model_registry import load_registry, register_model_version
from src.utils.settings import get_settings


def test_register_model_version(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 1, 1, 0])
    model = RandomForestClassifier(random_state=42).fit(X, y)

    model_path = settings.models_dir / "temp.joblib"
    joblib.dump(model, model_path)

    rec = register_model_version(model_path, accuracy=0.95, run_id="run_test")
    registry = load_registry()

    assert rec["version"] == 1
    assert registry["current_version"] == 1
    assert len(registry["history"]) == 1
