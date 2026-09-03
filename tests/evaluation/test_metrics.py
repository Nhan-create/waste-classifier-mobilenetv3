import numpy as np
import pytest

from src.data.schema import CLASS_NAMES
from src.evaluation.metrics import compute_multiclass_metrics


def test_perfect_predictions_have_ten_by_ten_metrics() -> None:
    targets = np.arange(10)
    probabilities = np.eye(10) * 0.9 + 0.01
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    result = compute_multiclass_metrics(targets, probabilities, CLASS_NAMES)

    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.weighted_f1 == 1.0
    assert len(result.per_class) == 10
    assert result.confusion_matrix.shape == (10, 10)
    assert result.normalized_confusion_matrix.shape == (10, 10)
    assert result.roc_auc_ovr_macro == pytest.approx(1.0)


def test_missing_class_returns_null_auc_and_complete_matrix() -> None:
    targets = np.arange(9)
    probabilities = np.eye(10)[:9]

    with pytest.warns(UserWarning, match="AUC"):
        result = compute_multiclass_metrics(targets, probabilities, CLASS_NAMES)

    assert result.roc_auc_ovr_macro is None
    assert result.confusion_matrix.shape == (10, 10)
    assert result.per_class[-1]["class_id"] == "trash"
    assert result.per_class[-1]["support"] == 0
