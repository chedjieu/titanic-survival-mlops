import pandas as pd
import pytest

from titanic_mlops.features.transform import TitanicFeatureTransformer, build_feature_pipeline
from titanic_mlops.features.validation import validate_features, validate_raw


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Pclass": 1,
                "Sex": "female",
                "Age": 29.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 211.34,
                "Embarked": "S",
                "Cabin": "B5",
                "Survived": 1,
            },
            {
                "Pclass": 3,
                "Sex": "male",
                "Age": None,
                "SibSp": 1,
                "Parch": 1,
                "Fare": None,
                "Embarked": None,
                "Cabin": None,
                "Survived": 0,
            },
        ]
    )


def test_validate_raw(raw_df):
    validated = validate_raw(raw_df)
    assert len(validated) == 2


def test_feature_transformer_imputes_and_engineers(raw_df):
    transformer = TitanicFeatureTransformer().fit(raw_df)
    features = transformer.transform(raw_df)
    validate_features(features)
    assert features.loc[1, "TravelAlone"] == 0
    assert features.loc[0, "TravelAlone"] == 1
    assert features["Age"].isna().sum() == 0
    assert features["Fare"].isna().sum() == 0


def test_pipeline_predict_shape(raw_df):
    y = raw_df["Survived"]
    X = raw_df.drop(columns=["Survived"])
    # duplicate rows so RF can train on tiny set
    X = pd.concat([X, X, X], ignore_index=True)
    y = pd.concat([y, y, y], ignore_index=True)
    pipe = build_feature_pipeline(n_estimators=10, random_state=0)
    pipe.fit(X, y)
    preds = pipe.predict(X.head(2))
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})
