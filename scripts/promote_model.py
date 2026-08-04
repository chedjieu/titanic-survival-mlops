"""Promote the latest registered model version to Production."""

from __future__ import annotations

import argparse
import logging
import sys

import mlflow
from mlflow.tracking import MlflowClient

from titanic_mlops.config import get_settings

logger = logging.getLogger(__name__)


def promote(version: str | None = None) -> str:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    if version is None:
        versions = client.search_model_versions(f"name='{settings.model_name}'")
        if not versions:
            raise RuntimeError(f"No versions found for {settings.model_name}")
        latest = max(versions, key=lambda v: int(v.version))
        version = str(latest.version)

    # Prefer MLflow aliases (MLflow 2.9+/3.x); fall back to stages when available.
    try:
        client.set_registered_model_alias(
            name=settings.model_name,
            alias="Production",
            version=version,
        )
        logger.info("Aliased %s v%s → @Production", settings.model_name, version)
    except Exception as alias_exc:  # noqa: BLE001
        logger.warning("Alias promotion failed (%s); trying stage API", alias_exc)
        client.transition_model_version_stage(
            name=settings.model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("Promoted %s v%s → Production stage", settings.model_name, version)
    return version


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=None)
    args = parser.parse_args(argv)
    promote(version=args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
