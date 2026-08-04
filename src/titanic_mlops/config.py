"""Shared configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_path: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    model_name: str
    min_f1_score: float
    random_state: int
    test_size: float
    kafka_bootstrap_servers: str
    kafka_topic_events: str
    kafka_topic_predictions: str
    kafka_topic_dlq: str
    kafka_group_id: str
    bento_service_url: str
    model_version: str
    drift_psi_threshold: float
    baseline_prediction_rate: float


def _resolve_mlflow_uri(root: Path) -> str:
    raw = os.getenv("MLFLOW_TRACKING_URI")
    default = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    if not raw:
        return default
    # MLflow 3 disallows the legacy file store unless explicitly opted in.
    if raw in {"./mlruns", "mlruns"} or raw.endswith("/mlruns") or raw.endswith("\\mlruns"):
        return default
    if raw.startswith("file:") and "mlruns" in raw:
        return default
    return raw


def get_settings() -> Settings:
    root = _project_root()
    default_data = root / "data" / "raw" / "titanic_data.csv"
    return Settings(
        project_root=root,
        data_path=Path(os.getenv("DATA_PATH", str(default_data))),
        mlflow_tracking_uri=_resolve_mlflow_uri(root),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "titanic-survival"),
        model_name=os.getenv("MODEL_NAME", "titanic-survival-classifier"),
        min_f1_score=float(os.getenv("MIN_F1_SCORE", "0.70")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        kafka_topic_events=os.getenv("KAFKA_TOPIC_EVENTS", "passenger.events"),
        kafka_topic_predictions=os.getenv("KAFKA_TOPIC_PREDICTIONS", "survival.predictions"),
        kafka_topic_dlq=os.getenv("KAFKA_TOPIC_DLQ", "survival.dlq"),
        kafka_group_id=os.getenv("KAFKA_GROUP_ID", "titanic-survival-consumer"),
        bento_service_url=os.getenv("BENTO_SERVICE_URL", "http://localhost:3000"),
        model_version=os.getenv("MODEL_VERSION", "Production"),
        drift_psi_threshold=float(os.getenv("DRIFT_PSI_THRESHOLD", "0.25")),
        baseline_prediction_rate=float(os.getenv("BASELINE_PREDICTION_RATE", "0.38")),
    )


FEATURE_COLUMNS = [
    "pclass_cat",
    "sex_cat",
    "Age",
    "Fare",
    "embarked_cat",
    "TravelAlone",
    "TravelTotal",
]

RAW_REQUIRED_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]
