import numpy as np

from titanic_mlops.monitoring.drift import evaluate_drift, population_stability_index


def test_psi_identical_is_near_zero():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=500)
    psi = population_stability_index(scores, scores)
    assert psi < 0.05


def test_evaluate_drift_flags_shift():
    baseline = [0.1] * 100
    live = [0.9] * 100
    result = evaluate_drift(live, baseline_probabilities=baseline, psi_threshold=0.1)
    assert result["drift_detected"] is True
