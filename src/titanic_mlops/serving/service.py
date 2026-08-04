"""BentoML SurvivalService — REST inference for Titanic survival."""

from __future__ import annotations

import logging
import os
from typing import Any

import bentoml
import pandas as pd
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PREDICTION_COUNTER = Counter(
    "titanic_predictions_total",
    "Total survival predictions served",
    ["survived"],
)
PREDICTION_LATENCY = Histogram(
    "titanic_prediction_latency_seconds",
    "Prediction latency in seconds",
)


class PassengerRequest(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str
    Age: float | None = None
    SibSp: int = 0
    Parch: int = 0
    Fare: float | None = None
    Embarked: str | None = "S"
    passenger_id: str | None = None


class PredictionResult(BaseModel):
    survived: int
    survival_probability: float
    model_version: str
    passenger_id: str | None = None


def _load_model() -> tuple[Any, str]:
    from titanic_mlops.serving.model_loader import load_pipeline

    return load_pipeline(os.getenv("MODEL_URI"))


@bentoml.service(
    name="survival_service",
    resources={"cpu": "1"},
    traffic={"timeout": 30},
)
class SurvivalService:
    def __init__(self) -> None:
        self.pipeline, self.model_version = _load_model()
        logger.info("SurvivalService ready with %s", self.model_version)

    @bentoml.api
    def healthz(self) -> dict[str, str]:
        return {"status": "ok", "model_version": self.model_version}

    @bentoml.api
    def predict(self, passenger: PassengerRequest) -> PredictionResult:
        with PREDICTION_LATENCY.time():
            frame = pd.DataFrame([passenger.model_dump(exclude={"passenger_id"})])
            proba = float(self.pipeline.predict_proba(frame)[0, 1])
            label = int(proba >= 0.5)
        PREDICTION_COUNTER.labels(survived=str(label)).inc()
        return PredictionResult(
            survived=label,
            survival_probability=proba,
            model_version=self.model_version,
            passenger_id=passenger.passenger_id,
        )

    @bentoml.api
    def predict_batch(self, passengers: list[PassengerRequest]) -> list[PredictionResult]:
        if not passengers:
            return []
        records = [p.model_dump(exclude={"passenger_id"}) for p in passengers]
        frame = pd.DataFrame(records)
        with PREDICTION_LATENCY.time():
            probas = self.pipeline.predict_proba(frame)[:, 1]
            labels = (probas >= 0.5).astype(int)
        results: list[PredictionResult] = []
        for passenger, label, proba in zip(passengers, labels, probas, strict=True):
            PREDICTION_COUNTER.labels(survived=str(int(label))).inc()
            results.append(
                PredictionResult(
                    survived=int(label),
                    survival_probability=float(proba),
                    model_version=self.model_version,
                    passenger_id=passenger.passenger_id,
                )
            )
        return results
