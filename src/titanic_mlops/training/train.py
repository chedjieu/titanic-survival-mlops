"""Train RandomForest pipeline, log to MLflow, and optionally register."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from titanic_mlops.config import get_settings
from titanic_mlops.features.transform import build_feature_pipeline, load_raw_dataframe
from titanic_mlops.features.validation import validate_raw
from titanic_mlops.training.evaluate import (
    compute_classification_metrics,
    confusion_matrix_dict,
    passes_quality_gate,
)

logger = logging.getLogger(__name__)


def train(
    data_path: Path | None = None,
    register: bool = True,
    tracking_uri: str | None = None,
) -> dict:
    settings = get_settings()
    path = Path(data_path or settings.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    resolved_uri = tracking_uri or settings.mlflow_tracking_uri
    os.environ["MLFLOW_TRACKING_URI"] = resolved_uri
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    raw = load_raw_dataframe(path)
    validate_raw(raw)
    if "Survived" not in raw.columns:
        raise ValueError("Training data must include Survived label column")

    labeled = raw.dropna(subset=["Survived"]).copy()
    labeled["Survived"] = labeled["Survived"].astype(int)
    y = labeled["Survived"]
    X = labeled.drop(columns=["Survived"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )

    pipeline = build_feature_pipeline(random_state=settings.random_state)
    params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "test_size": settings.test_size,
        "random_state": settings.random_state,
        "data_path": str(path),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    with mlflow.start_run(run_name="titanic-rf-train") as run:
        mlflow.log_params(params)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_classification_metrics(
            y_test.to_numpy(),
            y_pred,
            y_proba,
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(confusion_matrix_dict(y_test.to_numpy(), y_pred), "confusion_matrix.json")

        model_step = pipeline.named_steps["model"]
        transformed_names = [
            "Age",
            "Fare",
            "TravelTotal",
            "pclass_cat",
            "sex_cat",
            "embarked_cat",
            "TravelAlone",
        ]
        importances = {
            name: float(score)
            for name, score in zip(transformed_names, model_step.feature_importances_, strict=True)
        }
        mlflow.log_dict(importances, "feature_importances.json")

        gate_passed = passes_quality_gate(metrics, settings.min_f1_score)
        mlflow.set_tag("quality_gate_passed", str(gate_passed))
        mlflow.set_tag("min_f1_score", str(settings.min_f1_score))

        should_register = register and gate_passed
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=settings.model_name if should_register else None,
            input_example=X_train.head(3),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        artifact_dir = settings.project_root / "models"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = artifact_dir / "latest_pipeline.joblib"
        joblib.dump(pipeline, local_model_path)
        mlflow.log_artifact(str(local_model_path))

        result = {
            "run_id": run.info.run_id,
            "metrics": metrics,
            "quality_gate_passed": gate_passed,
            "registered": should_register,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "model_name": settings.model_name,
            "local_model_path": str(local_model_path),
        }
        result_path = artifact_dir / "latest_train_result.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Training complete: %s", json.dumps(result))

        if register and not gate_passed:
            logger.warning(
                "Quality gate failed (f1=%.4f < %.4f); model logged but not registered",
                metrics["f1"],
                settings.min_f1_score,
            )

        return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train Titanic survival model")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = train(
            data_path=args.data_path,
            register=not args.no_register,
            tracking_uri=args.tracking_uri,
        )
    except Exception:
        logger.exception("Training failed")
        return 1

    if args.fail_on_gate and not result["quality_gate_passed"]:
        logger.error("Exiting non-zero due to failed quality gate")
        return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
