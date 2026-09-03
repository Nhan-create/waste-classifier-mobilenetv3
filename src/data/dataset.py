"""Training-facing dataset discovery, counts, and class weights."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch

from .schema import CLASS_NAMES, VALID_IMAGE_EXTENSIONS, PipelineError


def discover_class_names(split_dir: Path) -> tuple[str, ...]:
    if not split_dir.is_dir():
        raise PipelineError(f"Dataset split directory not found: {split_dir}")
    actual = {path.name for path in split_dir.iterdir() if path.is_dir()}
    expected = set(CLASS_NAMES)
    if actual != expected:
        raise PipelineError(
            f"Invalid classes in {split_dir}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return CLASS_NAMES


def compute_class_counts(
    train_dir: Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, int]:
    discover_class_names(train_dir)
    counts: dict[str, int] = {}
    for class_name in class_names:
        class_root = train_dir / class_name
        counts[class_name] = sum(
            1
            for path in class_root.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
    return counts


def compute_class_weights(
    train_dir: Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, float]:
    counts = compute_class_counts(train_dir, class_names)
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise PipelineError(f"Cannot compute class weights; empty classes: {missing}")
    total = sum(counts.values())
    return {
        class_name: total / (len(class_names) * counts[class_name])
        for class_name in class_names
    }


def ordered_weight_tensor(
    weights_path: Path,
    class_names: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    try:
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PipelineError(f"Class weights not found: {weights_path}") from error
    expected = set(class_names)
    actual = set(payload)
    if actual != expected:
        raise PipelineError(
            f"Class-weight keys do not match classes; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    values = [float(payload[name]) for name in class_names]
    if any(value <= 0 for value in values):
        raise PipelineError("Every class weight must be positive")
    return torch.tensor(values, dtype=torch.float32, device=device)
