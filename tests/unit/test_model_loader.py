from titanic_mlops.serving.model_loader import load_pipeline


def test_load_pipeline_falls_back_to_local():
    pipeline, version = load_pipeline()
    assert hasattr(pipeline, "predict")
    assert "local:" in version or "models:/" in version
