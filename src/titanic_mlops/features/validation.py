"""Data quality gates for raw and engineered features."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from titanic_mlops.config import FEATURE_COLUMNS, RAW_REQUIRED_COLUMNS


class RawPassengerSchema(pa.DataFrameModel):
    Pclass: Series[int] = pa.Field(isin=[1, 2, 3])
    Sex: Series[str] = pa.Field(isin=["male", "female"])
    Age: Series[float] = pa.Field(nullable=True, ge=0, le=100)
    SibSp: Series[int] = pa.Field(ge=0)
    Parch: Series[int] = pa.Field(ge=0)
    Fare: Series[float] = pa.Field(nullable=True, ge=0)
    Embarked: Series[str] = pa.Field(nullable=True, isin=["C", "Q", "S"])

    class Config:
        strict = False
        coerce = True


class FeatureSchema(pa.DataFrameModel):
    pclass_cat: Series[int] = pa.Field(isin=[1, 2, 3])
    sex_cat: Series[int] = pa.Field(isin=[0, 1])
    Age: Series[float]
    Fare: Series[float]
    embarked_cat: Series[int] = pa.Field(isin=[0, 1, 2])
    TravelAlone: Series[int] = pa.Field(isin=[0, 1])
    TravelTotal: Series[float] = pa.Field(ge=1)

    class Config:
        strict = True
        coerce = True


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data missing required columns: {missing}")
    return RawPassengerSchema.validate(df, lazy=True)


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Feature frame missing columns: {missing}")
    return FeatureSchema.validate(df[FEATURE_COLUMNS], lazy=True)
