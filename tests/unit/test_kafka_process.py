import pandas as pd

from titanic_mlops.features.transform import build_feature_pipeline
from titanic_mlops.serving.kafka_runner import process_message


def test_process_message_returns_prediction(tmp_path):
    df = pd.DataFrame(
        [
            {
                "Pclass": 1,
                "Sex": "female",
                "Age": 29.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 100.0,
                "Embarked": "S",
                "Survived": 1,
            },
            {
                "Pclass": 3,
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked": "S",
                "Survived": 0,
            },
        ]
    )
    X = pd.concat([df.drop(columns=["Survived"])] * 5, ignore_index=True)
    y = pd.concat([df["Survived"]] * 5, ignore_index=True)
    pipe = build_feature_pipeline(n_estimators=5, random_state=0)
    pipe.fit(X, y)

    result = process_message(
        pipe,
        "test-version",
        {
            "event_id": "e1",
            "idempotency_key": "e1",
            "passenger_id": "p1",
            "Pclass": 1,
            "Sex": "female",
            "Age": 29.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 100.0,
            "Embarked": "S",
        },
    )
    assert result["model_version"] == "test-version"
    assert result["survived"] in (0, 1)
    assert 0.0 <= result["survival_probability"] <= 1.0
    assert result["idempotency_key"] == "e1"
