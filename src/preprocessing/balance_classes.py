"""Compute class weights without creating duplicate image files."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from src.data.dataset import compute_class_counts, compute_class_weights
from src.data.schema import CLASS_NAMES

__all__ = ["compute_class_counts", "compute_class_weights", "write_class_weights"]


def write_class_weights(
    train_dir: Path,
    output_path: Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, float]:
    weights = compute_class_weights(train_dir, class_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(weights, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return weights
