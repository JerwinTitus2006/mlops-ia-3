from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline.tasks import task_preprocess_data, task_train_baseline

with DAG(
    dag_id="ml_model_training",
    description="Preprocess data and train baseline model",
    schedule="15 */6 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "training"],
) as dag:
    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess_data,
    )

    train = PythonOperator(
        task_id="train_baseline",
        python_callable=task_train_baseline,
    )

    preprocess >> train
