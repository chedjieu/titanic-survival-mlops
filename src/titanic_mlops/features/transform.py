"""Shared feature engineering used by training and serving."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from titanic_mlops.config import FEATURE_COLUMNS

__all__ = [
    "FEATURE_COLUMNS",
    "TitanicFeatureTransformer",
    "build_feature_pipeline",
    "load_raw_dataframe",
]

SEX_MAP = {"female": 0, "male": 1}
EMBARKED_MAP = {"C": 0, "Q": 1, "S": 2}
CONTINUOUS = ["Age", "Fare", "TravelTotal"]


class TitanicFeatureTransformer(BaseEstimator, TransformerMixin):
    """Clean and engineer Titanic features with train-time imputation stats."""

    def __init__(self) -> None:
        self.age_median_: float | None = None
        self.fare_median_: float | None = None

    def fit(self, X: pd.DataFrame, y: Iterable | None = None):  # noqa: ARG002
        frame = self._as_frame(X)
        self.age_median_ = float(frame["Age"].median(skipna=True))
        self.fare_median_ = float(frame["Fare"].median(skipna=True))
        return self

    def transform(self, X: pd.DataFrame | list[dict] | dict) -> pd.DataFrame:
        frame = self._as_frame(X).copy()
        if self.age_median_ is None or self.fare_median_ is None:
            raise RuntimeError("TitanicFeatureTransformer must be fit before transform")

        if "Cabin" in frame.columns:
            frame = frame.drop(columns=["Cabin"])

        frame["Age"] = frame["Age"].fillna(self.age_median_)
        frame["Fare"] = frame["Fare"].fillna(self.fare_median_)
        frame["Embarked"] = frame["Embarked"].fillna("S")

        travel_group = frame["SibSp"].fillna(0) + frame["Parch"].fillna(0)
        frame["TravelAlone"] = np.where(travel_group > 0, 0, 1).astype(int)
        frame["TravelTotal"] = (travel_group + 1).astype(float)

        frame["pclass_cat"] = frame["Pclass"].astype(int)
        # Keep label encoding consistent with notebook semantics (sorted unique fit order
        # for Sex female/male and Embarked C/Q/S matches LabelEncoder on full data).
        frame["sex_cat"] = frame["Sex"].map(SEX_MAP).astype(int)
        embarked = frame["Embarked"].map(EMBARKED_MAP).fillna(EMBARKED_MAP["S"])
        frame["embarked_cat"] = embarked.astype(int)

        for col in CONTINUOUS:
            frame[col] = frame[col].astype("float64")

        return frame[FEATURE_COLUMNS]

    @staticmethod
    def _as_frame(X: pd.DataFrame | list[dict] | dict) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, dict):
            return pd.DataFrame([X])
        return pd.DataFrame(list(X))


def build_feature_pipeline(
    n_estimators: int = 200,
    random_state: int = 42,
) -> Pipeline:
    """Sklearn pipeline: feature transform → scale continuous → RandomForest."""
    feature_transformer = TitanicFeatureTransformer()
    scaler = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), CONTINUOUS),
            (
                "passthrough",
                "passthrough",
                ["pclass_cat", "sex_cat", "embarked_cat", "TravelAlone"],
            ),
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("features", feature_transformer),
            ("scale", scaler),
            ("model", model),
        ]
    )


def load_raw_dataframe(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
