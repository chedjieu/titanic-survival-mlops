"""Airflow DAG: validate → train → quality gate → register metadata."""

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

PROJECT_ROOT = os.environ.get("TITANIC_PROJECT_ROOT", "/opt/titanic-survival-mlops")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, str(Path(PROJECT_ROOT) / "src"))


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _check_gate(**context) -> str:
    result_path = Path(PROJECT_ROOT) / "models" / "latest_train_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    context["ti"].xcom_push(key="train_result", value=result)
    if result.get("quality_gate_passed"):
        return "register_success"
    return "register_skipped"


with DAG(
    dag_id="train_and_register",
    default_args=default_args,
    description="Train Titanic model, gate on F1, register in MLflow",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["titanic", "mlflow", "training"],
) as dag:
    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -c \"from pathlib import Path; from titanic_mlops.features import "
            "load_raw_dataframe, validate_raw; from titanic_mlops.config import get_settings; "
            "s=get_settings(); validate_raw(load_raw_dataframe(s.data_path)); print('ok')\""
        ),
        env={
            "PYTHONPATH": f"{PROJECT_ROOT}/src",
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            "DATA_PATH": f"{PROJECT_ROOT}/data/raw/titanic_data.csv",
        },
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m titanic_mlops.training.train"
        ),
        env={
            "PYTHONPATH": f"{PROJECT_ROOT}/src",
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            "DATA_PATH": f"{PROJECT_ROOT}/data/raw/titanic_data.csv",
            "MIN_F1_SCORE": os.environ.get("MIN_F1_SCORE", "0.70"),
        },
    )

    gate = BranchPythonOperator(
        task_id="evaluate_quality_gate",
        python_callable=_check_gate,
    )

    register_success = EmptyOperator(task_id="register_success")
    register_skipped = EmptyOperator(task_id="register_skipped")
    done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")

    validate_data >> train_model >> gate
    gate >> [register_success, register_skipped] >> done
