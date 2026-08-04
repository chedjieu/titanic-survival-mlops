"""Load champion model from local artifact or MLflow registry."""

from __future__ import annotations

import logging
from typing import Any

import joblib
import mlflow

from titanic_mlops.config import get_settings

logger = logging.getLogger(__name__)


def load_pipeline(model_uri: str | None = None) -> tuple[Any, str]:
    """Return (pipeline, version_label)."""
    settings = get_settings()
    local_path = settings.project_root / "models" / "latest_pipeline.joblib"

    if model_uri:
        pipeline = mlflow.sklearn.load_model(model_uri)
        return pipeline, model_uri

    stage_or_version = settings.model_version
    # Support alias (@Production), stage/version path, and numeric versions.
    if stage_or_version.startswith("@"):
        candidates = [f"models:/{settings.model_name}{stage_or_version}"]
    elif stage_or_version in {"Production", "Staging", "Archived", "None"}:
        candidates = [
            f"models:/{settings.model_name}@{stage_or_version}",
            f"models:/{settings.model_name}/{stage_or_version}",
        ]
    else:
        candidates = [f"models:/{settings.model_name}/{stage_or_version}"]

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    last_exc: Exception | None = None
    for registry_uri in candidates:
        try:
            pipeline = mlflow.sklearn.load_model(registry_uri)
            logger.info("Loaded model from registry %s", registry_uri)
            return pipeline, registry_uri
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Registry load failed for %s (%s)", registry_uri, exc)

    logger.warning("All registry loads failed (%s); falling back to %s", last_exc, local_path)

    if not local_path.exists():
        raise FileNotFoundError(
            f"No model available. Train first or set a valid MLflow model. Tried {registry_uri}"
        )
    pipeline = joblib.load(local_path)
    return pipeline, f"local:{local_path}"
