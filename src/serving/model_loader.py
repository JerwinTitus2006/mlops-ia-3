from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.utils.settings import get_settings


class ModelLoader:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = None
        self.model_version: int | None = None
        self.registry_mtime: float = -1.0

    def _registry_path(self) -> Path:
        return self.settings.models_dir / "registry.json"

    def _current_model_path(self) -> Path:
        return self.settings.models_dir / "current_model.joblib"

    def refresh_if_needed(self) -> None:
        registry_path = self._registry_path()
        if not registry_path.exists():
            return

        current_mtime = registry_path.stat().st_mtime
        if current_mtime == self.registry_mtime and self.model is not None:
            return

        with registry_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)

        model_path = self._current_model_path()
        if model_path.exists():
            self.model = joblib.load(model_path)
            self.model_version = registry.get("current_version")
            self.registry_mtime = current_mtime

    def predict(self, data: Any):
        self.refresh_if_needed()
        if self.model is None:
            raise RuntimeError("No model is currently available")
        return self.model.predict(data)

    def health(self) -> dict:
        self.refresh_if_needed()
        return {
            "model_loaded": self.model is not None,
            "model_version": self.model_version,
        }
