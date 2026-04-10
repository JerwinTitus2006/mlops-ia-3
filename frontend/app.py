from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="Self-Healing ML Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


@st.cache_data(ttl=5)
def fetch_health() -> dict[str, Any]:
    return requests.get(f"{API_BASE_URL}/health", timeout=10).json()


@st.cache_data(ttl=5)
def fetch_metrics_text() -> str:
    return requests.get(f"{API_BASE_URL}/metrics", timeout=10).text


@st.cache_data(ttl=5)
def fetch_drift_report() -> dict[str, Any]:
    drift_path = MODELS_DIR / "drift_report.json"
    if drift_path.exists():
        return json.loads(drift_path.read_text(encoding="utf-8"))
    return {}


@st.cache_data(ttl=5)
def fetch_model_registry() -> dict[str, Any]:
    registry_path = MODELS_DIR / "registry.json"
    if registry_path.exists():
        return json.loads(registry_path.read_text(encoding="utf-8"))
    return {"current_version": None, "history": []}


st.title("Self-Healing ML Pipeline Dashboard")
st.caption("Live view of model health, drift, predictions, and version history.")

health_col, drift_col, version_col, alert_col = st.columns(4)

try:
    health = fetch_health()
except Exception as exc:  # pragma: no cover - dashboard runtime guard
    st.error(f"Could not reach API at {API_BASE_URL}: {exc}")
    st.stop()

registry = fetch_model_registry()
drift = fetch_drift_report()

health_col.metric("Model Loaded", "Yes" if health.get("model_loaded") else "No")
drift_col.metric("Drift Score", f'{drift.get("drift_score", 0.0):.3f}')
version_col.metric("Model Version", health.get("model_version", "N/A"))
alert_col.metric("Drift Triggered", "Yes" if drift.get("drift_detected") else "No")

left, right = st.columns([1.05, 1.0])

with left:
    st.subheader("Make a Prediction")
    with st.form("predict_form"):
        row_1 = st.text_input("Row 1 (comma-separated iris features)", "5.1,3.5,1.4,0.2")
        row_2 = st.text_input("Row 2 (comma-separated iris features)", "6.2,2.8,4.8,1.8")
        submitted = st.form_submit_button("Predict")

    if submitted:
        def parse_row(raw: str) -> list[float]:
            values = [float(item.strip()) for item in raw.split(",") if item.strip()]
            if len(values) != 4:
                raise ValueError("Each row must contain exactly 4 numeric values")
            return values

        try:
            payload = {"rows": [parse_row(row_1), parse_row(row_2)]}
            response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            st.success("Prediction completed")
            st.json(result)
        except Exception as exc:  # pragma: no cover - dashboard runtime guard
            st.error(f"Prediction failed: {exc}")

with right:
    st.subheader("Drift & Version Overview")

    if drift:
        drift_df = pd.DataFrame(
            {
                "feature": list(drift.get("feature_scores", {}).keys()),
                "psi": list(drift.get("feature_scores", {}).values()),
            }
        )
        if not drift_df.empty:
            fig = px.bar(drift_df, x="feature", y="psi", color="psi", title="PSI per Feature")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No drift report found yet. Run the self-healing cycle or wait for Airflow.")

    if registry.get("history"):
        history_df = pd.DataFrame(registry["history"])
        st.dataframe(history_df[["version", "accuracy", "run_id"]], use_container_width=True)
    else:
        st.info("No model registry history found yet.")

st.subheader("Prometheus Metrics Preview")
metrics_text = fetch_metrics_text()
st.code(
    "\n".join(
        line for line in metrics_text.splitlines()
        if line.startswith("ml_api_request_total")
        or line.startswith("ml_api_request_latency_seconds_count")
        or line.startswith("ml_drift_score")
        or line.startswith("ml_prediction_class_count")
    ) or "No custom metrics found yet.",
    language="text",
)

st.subheader("Latest Artifacts")
artifact_cols = st.columns(3)
artifact_cols[0].write("Raw data")
artifact_cols[0].code(str(DATA_RAW_DIR), language="text")
artifact_cols[1].write("Model store")
artifact_cols[1].code(str(MODELS_DIR), language="text")
artifact_cols[2].write("API endpoint")
artifact_cols[2].code(API_BASE_URL, language="text")
