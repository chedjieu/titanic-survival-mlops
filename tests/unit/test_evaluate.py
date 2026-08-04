import numpy as np

from titanic_mlops.training.evaluate import (
    compute_classification_metrics,
    passes_quality_gate,
)


def test_metrics_and_gate():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert passes_quality_gate({"f1": 0.75}, 0.7)
    assert not passes_quality_gate({"f1": 0.5}, 0.7)
