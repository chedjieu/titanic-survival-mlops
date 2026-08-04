"""Request/response schemas for REST and Kafka payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PassengerFeatures(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str
    Age: float | None = None
    SibSp: int = 0
    Parch: int = 0
    Fare: float | None = None
    Embarked: str | None = "S"
    passenger_id: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "Pclass": self.Pclass,
            "Sex": self.Sex,
            "Age": self.Age,
            "SibSp": self.SibSp,
            "Parch": self.Parch,
            "Fare": self.Fare,
            "Embarked": self.Embarked or "S",
        }


class PredictionResponse(BaseModel):
    survived: int
    survival_probability: float
    model_version: str
    passenger_id: str | None = None
