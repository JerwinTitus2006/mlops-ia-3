from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from src.pipeline.tasks import task_retrain_decision, task_retrain_if_needed


def _branch(**context):
    ti = context["ti"]
    decision = ti.xcom_pull(task_ids="retrain_decision")
    if decision and decision.get("trigger_retraining"):
        return "retrain"
    return "skip"


with DAG(
    dag_id="ml_retraining_trigger",
    description="Trigger self-healing retraining when drift/performance issues are found",
    schedule="*/20 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "self-healing"],
) as dag:
    decide = PythonOperator(
        task_id="retrain_decision",
        python_callable=task_retrain_decision,
    )

    branch = BranchPythonOperator(
        task_id="branch",
        python_callable=_branch,
    )

    retrain = PythonOperator(
        task_id="retrain",
        python_callable=task_retrain_if_needed,
    )

    skip = EmptyOperator(task_id="skip")
    done = EmptyOperator(task_id="done")

    decide >> branch
    branch >> retrain >> done
    branch >> skip >> done
