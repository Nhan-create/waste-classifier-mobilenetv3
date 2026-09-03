"""Pure ten-class metric calculations."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricBundle:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: list[dict[str, float | int | str]]
    confusion_matrix: np.ndarray
    normalized_confusion_matrix: np.ndarray
    roc_auc_ovr_macro: float | None


def compute_multiclass_metrics(
    targets: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> MetricBundle:
    targets_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    class_count = len(class_names)
    if targets_array.ndim != 1 or targets_array.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional array")
    if probability_array.shape != (targets_array.size, class_count):
        raise ValueError(
            f"probabilities must have shape {(targets_array.size, class_count)}, "
            f"received {probability_array.shape}"
        )
    labels = np.arange(class_count)
    predictions = probability_array.argmax(axis=1)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        targets_array,
        predictions,
        labels=labels,
        zero_division=0,
    )
    raw_confusion = confusion_matrix(targets_array, predictions, labels=labels)
    row_totals = raw_confusion.sum(axis=1, keepdims=True)
    normalized_confusion = np.divide(
        raw_confusion,
        row_totals,
        out=np.zeros_like(raw_confusion, dtype=np.float64),
        where=row_totals != 0,
    )
    if set(np.unique(targets_array)) == set(labels):
        auc = float(
            roc_auc_score(
                targets_array,
                probability_array,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    else:
        warnings.warn(
            "One-vs-rest macro AUC is unavailable because the test set "
            "does not contain every class.",
            UserWarning,
            stacklevel=2,
        )
        auc = None
    per_class = [
        {
            "class_id": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    ]
    return MetricBundle(
        accuracy=float(accuracy_score(targets_array, predictions)),
        macro_f1=float(
            f1_score(
                targets_array,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        weighted_f1=float(
            f1_score(
                targets_array,
                predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        per_class=per_class,
        confusion_matrix=raw_confusion,
        normalized_confusion_matrix=normalized_confusion,
        roc_auc_ovr_macro=auc,
    )
