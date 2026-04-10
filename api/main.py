from __future__ import annotations

import logging
from collections import Counter

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import make_asgi_app

from src.data.ingestion import FEATURE_COLUMNS
from src.monitoring.metrics import (
    DRIFT_SCORE_GAUGE,
    PREDICTION_CLASS_GAUGE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from src.serving.model_loader import ModelLoader
from src.training.model_registry import rollback_to_version
from src.utils.logging_utils import configure_logging
from src.utils.settings import get_settings

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Self-Healing ML Serving API", version="1.0.0")
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

loader = ModelLoader()


class PredictRequest(BaseModel):
    rows: list[list[float]] = Field(..., description="Batch of feature rows in iris feature order")


class RollbackRequest(BaseModel):
    version: int


@app.get("/health")
def health() -> dict:
    REQUEST_COUNT.labels(endpoint="health").inc()
    with REQUEST_LATENCY.labels(endpoint="health").time():
        status = loader.health()
        return {"status": "ok", **status}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    REQUEST_COUNT.labels(endpoint="predict").inc()
    with REQUEST_LATENCY.labels(endpoint="predict").time():
        if not payload.rows:
            raise HTTPException(status_code=400, detail="rows must not be empty")

        if any(len(row) != len(FEATURE_COLUMNS) for row in payload.rows):
            raise HTTPException(
                status_code=400,
                detail=f"Each row must have exactly {len(FEATURE_COLUMNS)} values",
            )

        df = pd.DataFrame(payload.rows, columns=FEATURE_COLUMNS)
        try:
            preds = loader.predict(df).tolist()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

        class_counts = Counter(str(p) for p in preds)
        for label, count in class_counts.items():
            PREDICTION_CLASS_GAUGE.labels(class_label=label).set(count)

        drift_path = get_settings().models_dir / "drift_report.json"
        if drift_path.exists():
            import json

            with drift_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
            DRIFT_SCORE_GAUGE.set(float(report.get("drift_score", 0.0)))

        return {
            "predictions": preds,
            "model_version": loader.health().get("model_version"),
        }


@app.post("/rollback")
def rollback(payload: RollbackRequest) -> dict:
    REQUEST_COUNT.labels(endpoint="rollback").inc()
    with REQUEST_LATENCY.labels(endpoint="rollback").time():
        try:
            version_item = rollback_to_version(payload.version)
            loader.refresh_if_needed()
            logger.warning("Rolled back model to version=%s", payload.version)
            return {
                "status": "rolled_back",
                "version": version_item["version"],
                "accuracy": version_item["accuracy"],
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
