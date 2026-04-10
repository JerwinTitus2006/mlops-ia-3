from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline.tasks import task_detect_drift

with DAG(
    dag_id="ml_drift_detection",
    description="Compute PSI data drift scores",
    schedule="*/20 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "drift"],
) as dag:
    drift = PythonOperator(
        task_id="detect_drift",
        python_callable=task_detect_drift,
    )
