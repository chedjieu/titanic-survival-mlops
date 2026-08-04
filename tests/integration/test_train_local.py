from pathlib import Path

import mlflow

from titanic_mlops.training.train import train


def test_train_writes_artifacts(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    data_path = root / "data" / "raw" / "titanic_data.csv"
    tracking = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    monkeypatch.setenv("MIN_F1_SCORE", "0.50")

    result = train(data_path=data_path, register=False, tracking_uri=tracking)
    assert result["quality_gate_passed"] is True
    assert Path(result["local_model_path"]).exists()
    assert "f1" in result["metrics"]

    mlflow.set_tracking_uri(tracking)
    run = mlflow.get_run(result["run_id"])
    assert run.info.run_id == result["run_id"]
