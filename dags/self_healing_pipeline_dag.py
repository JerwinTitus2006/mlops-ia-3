from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from src.pipeline.tasks import (
    task_check_performance,
    task_detect_drift,
    task_ingest_data,
    task_preprocess_data,
    task_retrain_decision,
    task_retrain_if_needed,
    task_train_baseline,
)


def _branch_on_retrain(**context):
    ti = context["ti"]
    decision = ti.xcom_pull(task_ids="retraining_decision")
    if decision and decision.get("trigger_retraining"):
        return "retrain_and_deploy"
    return "no_retraining_needed"


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="self_healing_ml_pipeline",
    default_args=default_args,
    description="End-to-end self-healing ML pipeline",
    schedule="*/15 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "self-healing", "ml"],
) as dag:
    ingest_data = PythonOperator(
        task_id="ingest_data",
        python_callable=task_ingest_data,
    )

    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess_data,
    )

    train_baseline = PythonOperator(
        task_id="train_baseline",
        python_callable=task_train_baseline,
    )

    detect_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=task_detect_drift,
    )

    check_performance = PythonOperator(
        task_id="check_performance",
        python_callable=task_check_performance,
    )

    retraining_decision = PythonOperator(
        task_id="retraining_decision",
        python_callable=task_retrain_decision,
    )

    branch = BranchPythonOperator(
        task_id="branch_retraining",
        python_callable=_branch_on_retrain,
    )

    retrain_and_deploy = PythonOperator(
        task_id="retrain_and_deploy",
        python_callable=task_retrain_if_needed,
    )

    no_retraining_needed = EmptyOperator(task_id="no_retraining_needed")
    done = EmptyOperator(task_id="pipeline_done")

    ingest_data >> preprocess_data >> train_baseline
    train_baseline >> [detect_drift, check_performance]
    [detect_drift, check_performance] >> retraining_decision >> branch
    branch >> retrain_and_deploy >> done
    branch >> no_retraining_needed >> done
