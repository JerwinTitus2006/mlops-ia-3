from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.utils.settings import get_settings


REGISTRY_FILE = "registry.json"


def _registry_path() -> Path:
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return settings.models_dir / REGISTRY_FILE


def load_registry() -> dict[str, Any]:
    path = _registry_path()
    if not path.exists():
        return {"current_version": None, "history": []}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict[str, Any]) -> None:
    path = _registry_path()
    with path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def next_version() -> int:
    registry = load_registry()
    history = registry.get("history", [])
    if not history:
        return 1
    return max(item["version"] for item in history) + 1


def register_model_version(model_path: Path, accuracy: float, run_id: str) -> dict[str, Any]:
    settings = get_settings()
    registry = load_registry()
    version = next_version()

    versioned_path = settings.models_dir / f"model_v{version}.joblib"
    shutil.copy2(model_path, versioned_path)

    record = {
        "version": version,
        "path": str(versioned_path),
        "accuracy": accuracy,
        "run_id": run_id,
    }

    registry.setdefault("history", []).append(record)
    registry["current_version"] = version
    save_registry(registry)

    current_model_path = settings.models_dir / "current_model.joblib"
    shutil.copy2(versioned_path, current_model_path)

    return record


def rollback_to_version(version: int) -> dict[str, Any]:
    settings = get_settings()
    registry = load_registry()
    version_item = next((h for h in registry.get("history", []) if h["version"] == version), None)
    if not version_item:
        raise ValueError(f"Version {version} not found in registry")

    source = Path(version_item["path"])
    target = settings.models_dir / "current_model.joblib"
    shutil.copy2(source, target)

    registry["current_version"] = version
    save_registry(registry)
    return version_item
