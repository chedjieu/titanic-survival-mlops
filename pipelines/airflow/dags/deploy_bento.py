"""Airflow DAG: build/push Bento image from Production MLflow model."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = os.environ.get("TITANIC_PROJECT_ROOT", "/opt/titanic-survival-mlops")

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="deploy_bento",
    default_args=default_args,
    description="Build and roll out BentoML SurvivalService",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["titanic", "bentoml", "deploy"],
) as dag:
    build_bento = BashOperator(
        task_id="build_bento",
        bash_command=(
            f"cd {PROJECT_ROOT}/src/titanic_mlops/serving && "
            "bentoml build -f bentofile.yaml"
        ),
        env={
            "PYTHONPATH": f"{PROJECT_ROOT}/src",
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            "MODEL_VERSION": "Production",
        },
    )

    containerize = BashOperator(
        task_id="containerize_bento",
        bash_command=(
            "bentoml containerize survival_service:latest "
            "-t titanic-survival-service:latest"
        ),
    )

    rollout = BashOperator(
        task_id="rollout_compose",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "docker compose -f infra/docker-compose.yml up -d --no-deps --build bento"
        ),
    )

    health_check = BashOperator(
        task_id="health_check",
        bash_command=(
            "curl -sf http://bento:3000/healthz || "
            "curl -sf http://localhost:3000/healthz"
        ),
        retries=5,
        retry_delay=timedelta(seconds=15),
    )

    build_bento >> containerize >> rollout >> health_check
