from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline.tasks import task_ingest_data

with DAG(
    dag_id="ml_data_ingestion",
    description="Ingest raw baseline and incoming datasets",
    schedule="0 */6 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "ingestion"],
) as dag:
    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=task_ingest_data,
    )
