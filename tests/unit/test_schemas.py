from titanic_mlops.serving.schemas import PassengerFeatures, PredictionResponse


def test_passenger_to_record():
    p = PassengerFeatures(Pclass=2, Sex="male", Age=30, Fare=10.0, Embarked="C")
    record = p.to_record()
    assert record["Pclass"] == 2
    assert record["Sex"] == "male"


def test_prediction_response():
    r = PredictionResponse(
        survived=1,
        survival_probability=0.82,
        model_version="models:/titanic-survival-classifier/Production",
        passenger_id="42",
    )
    assert r.survived == 1
