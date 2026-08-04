"""Prediction-rate drift monitoring using PSI against a baseline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from titanic_mlops.config import get_settings

logger = logging.getLogger(__name__)


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """PSI between two score distributions in [0, 1]."""
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.quantile(expected, quantiles)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
    actual_counts = np.histogram(actual, bins=breakpoints)[0].astype(float)
    expected_pct = expected_counts / max(expected_counts.sum(), eps) + eps
    actual_pct = actual_counts / max(actual_counts.sum(), eps) + eps
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def prediction_rate_delta(actual_rate: float, baseline_rate: float) -> float:
    return abs(actual_rate - baseline_rate)


def evaluate_drift(
    probabilities: list[float],
    baseline_probabilities: list[float] | None = None,
    baseline_rate: float | None = None,
    psi_threshold: float | None = None,
) -> dict:
    settings = get_settings()
    psi_threshold = psi_threshold if psi_threshold is not None else settings.drift_psi_threshold
    baseline_rate = (
        baseline_rate if baseline_rate is not None else settings.baseline_prediction_rate
    )
    actual = np.asarray(probabilities, dtype=float)
    if actual.size == 0:
        raise ValueError("No probabilities provided for drift evaluation")

    actual_rate = float((actual >= 0.5).mean())
    rate_delta = prediction_rate_delta(actual_rate, baseline_rate)

    if baseline_probabilities:
        expected = np.asarray(baseline_probabilities, dtype=float)
        psi = population_stability_index(expected, actual)
    else:
        # Synthetic baseline around historical positive rate
        expected = np.random.default_rng(42).normal(loc=baseline_rate, scale=0.15, size=1000)
        expected = np.clip(expected, 0, 1)
        psi = population_stability_index(expected, actual)

    drifted = psi >= psi_threshold or rate_delta >= psi_threshold
    return {
        "psi": psi,
        "psi_threshold": psi_threshold,
        "actual_positive_rate": actual_rate,
        "baseline_positive_rate": baseline_rate,
        "rate_delta": rate_delta,
        "drift_detected": drifted,
        "n_predictions": int(actual.size),
    }


def load_probabilities_from_jsonl(path: Path) -> list[float]:
    probs: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        probs.append(float(record["survival_probability"]))
    return probs


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate prediction drift")
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        required=True,
        help="JSONL file with survival_probability fields",
    )
    parser.add_argument("--baseline-jsonl", type=Path, default=None)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args(argv)

    probs = load_probabilities_from_jsonl(args.predictions_jsonl)
    baseline = (
        load_probabilities_from_jsonl(args.baseline_jsonl) if args.baseline_jsonl else None
    )
    result = evaluate_drift(probs, baseline_probabilities=baseline)
    print(json.dumps(result, indent=2))

    out = get_settings().project_root / "models" / "latest_drift_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.fail_on_drift and result["drift_detected"]:
        logger.error("Drift detected")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
