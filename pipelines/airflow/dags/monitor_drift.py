"""Airflow DAG: consume prediction snapshot, evaluate drift, optionally retrain."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

PROJECT_ROOT = os.environ.get("TITANIC_PROJECT_ROOT", "/opt/titanic-survival-mlops")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, str(Path(PROJECT_ROOT) / "src"))

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _branch_on_drift(**context) -> str:
    result_path = Path(PROJECT_ROOT) / "models" / "latest_drift_result.json"
    if not result_path.exists():
        return "no_drift"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    context["ti"].xcom_push(key="drift_result", value=result)
    return "trigger_retrain" if result.get("drift_detected") else "no_drift"


with DAG(
    dag_id="monitor_drift",
    default_args=default_args,
    description="Monitor prediction drift and trigger retrain",
    schedule_interval="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["titanic", "monitoring", "kafka"],
) as dag:
    collect_predictions = BashOperator(
        task_id="collect_predictions",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python scripts/collect_kafka_predictions.py "
            "--output models/live_predictions.jsonl --max-messages 200 --timeout 20"
        ),
        env={
            "PYTHONPATH": f"{PROJECT_ROOT}/src",
            "KAFKA_BOOTSTRAP_SERVERS": os.environ.get(
                "KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"
            ),
        },
    )

    evaluate_drift = BashOperator(
        task_id="evaluate_drift",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m titanic_mlops.monitoring.drift "
            "--predictions-jsonl models/live_predictions.jsonl"
        ),
        env={"PYTHONPATH": f"{PROJECT_ROOT}/src"},
    )

    branch = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=_branch_on_drift,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="train_and_register",
        wait_for_completion=False,
    )
    no_drift = EmptyOperator(task_id="no_drift")

    collect_predictions >> evaluate_drift >> branch
    branch >> [trigger_retrain, no_drift]
