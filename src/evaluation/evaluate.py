"""Evaluate one selected checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.data.schema import PipelineError
from src.evaluation.metrics import MetricBundle, compute_multiclass_metrics
from src.training.checkpoint import (
    CheckpointCompatibilityError,
    build_model_from_checkpoint,
    load_checkpoint,
)


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    metrics: MetricBundle


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _evaluation_transform(
    input_size: int,
    normalization: dict[str, list[float]],
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalization["mean"],
                std=normalization["std"],
            ),
        ]
    )


def _validate_test_class_directories(
    test_root: Path,
    class_names: tuple[str, ...],
) -> None:
    if not test_root.is_dir():
        raise PipelineError(f"Test directory not found: {test_root}")
    actual = {path.name for path in test_root.iterdir() if path.is_dir()}
    expected = set(class_names)
    if actual != expected:
        raise PipelineError(
            f"Test classes do not match checkpoint; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _write_metrics_json(
    output_path: Path,
    metrics: MetricBundle,
    class_names: tuple[str, ...],
) -> None:
    payload = {
        "class_names": list(class_names),
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "roc_auc_ovr_macro": metrics.roc_auc_ovr_macro,
        "confusion_matrix_raw": metrics.confusion_matrix.tolist(),
        "confusion_matrix_normalized": (
            metrics.normalized_confusion_matrix.tolist()
        ),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_per_class_csv(
    output_path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("class_id", "precision", "recall", "f1", "support"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_plot(
    path: Path,
    matrix: np.ndarray,
    class_names: tuple[str, ...],
    *,
    normalized: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1%" if normalized else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(
        "Normalized confusion matrix" if normalized else "Raw confusion matrix"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_checkpoint(
    checkpoint_path: Path,
    test_root: Path,
    output_dir: Path,
    *,
    device: str = "auto",
    batch_size: int = 32,
    num_workers: int = 0,
) -> EvaluationResult:
    resolved_device = _resolve_device(device)
    loaded = load_checkpoint(checkpoint_path, resolved_device)
    _validate_test_class_directories(test_root, loaded.metadata.class_names)
    dataset = ImageFolder(
        test_root,
        transform=_evaluation_transform(
            loaded.metadata.input_size,
            loaded.metadata.normalization,
        ),
        allow_empty=True,
    )
    if tuple(dataset.classes) != loaded.metadata.class_names:
        raise PipelineError(
            f"ImageFolder class order {tuple(dataset.classes)} does not match "
            f"checkpoint {loaded.metadata.class_names}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
    )
    model = build_model_from_checkpoint(loaded, resolved_device)
    probability_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, targets in loader:
            logits = model(inputs.to(resolved_device))
            probability_batches.append(
                torch.softmax(logits, dim=1).cpu().numpy()
            )
            target_batches.append(targets.numpy())
    if not target_batches:
        raise PipelineError(f"Test directory contains no images: {test_root}")
    probabilities = np.concatenate(probability_batches)
    targets = np.concatenate(target_batches)
    metrics = compute_multiclass_metrics(
        targets,
        probabilities,
        loaded.metadata.class_names,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_json(
        output_dir / "metrics.json",
        metrics,
        loaded.metadata.class_names,
    )
    _write_per_class_csv(output_dir / "per_class_metrics.csv", metrics.per_class)
    _write_confusion_plot(
        output_dir / "confusion_matrix_raw.png",
        metrics.confusion_matrix,
        loaded.metadata.class_names,
        normalized=False,
    )
    _write_confusion_plot(
        output_dir / "confusion_matrix_normalized.png",
        metrics.normalized_confusion_matrix,
        loaded.metadata.class_names,
        normalized=True,
    )
    return EvaluationResult(output_dir=output_dir, metrics=metrics)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        result = evaluate_checkpoint(
            arguments.checkpoint,
            arguments.test_root,
            arguments.output_dir,
            device=arguments.device,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
        )
    except (CheckpointCompatibilityError, PipelineError, RuntimeError, ValueError) as error:
        print(f"Evaluation failed: {error}")
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "accuracy": result.metrics.accuracy,
                "macro_f1": result.metrics.macro_f1,
                "weighted_f1": result.metrics.weighted_f1,
                "roc_auc_ovr_macro": result.metrics.roc_auc_ovr_macro,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
