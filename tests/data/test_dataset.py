import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.dataset import (
    compute_class_counts,
    compute_class_weights,
    discover_class_names,
    ordered_weight_tensor,
)
from src.data.schema import CLASS_NAMES, PipelineError


def create_class_tree(root: Path, battery_count: int = 1) -> None:
    for class_name in CLASS_NAMES:
        count = battery_count if class_name == "battery" else 1
        for index in range(count):
            path = root / class_name / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 12), class_name == "battery" and "red" or "blue").save(path)


def test_class_discovery_requires_exact_ten_class_set(tmp_path: Path) -> None:
    create_class_tree(tmp_path)
    assert discover_class_names(tmp_path) == CLASS_NAMES

    (tmp_path / "trash" / "0.png").unlink()
    (tmp_path / "trash").rmdir()
    with pytest.raises(PipelineError, match="trash"):
        discover_class_names(tmp_path)


def test_counts_and_weights_follow_canonical_order(tmp_path: Path) -> None:
    create_class_tree(tmp_path, battery_count=2)

    counts = compute_class_counts(tmp_path, CLASS_NAMES)
    weights = compute_class_weights(tmp_path, CLASS_NAMES)

    assert list(counts) == list(CLASS_NAMES)
    assert counts["battery"] == 2
    assert sum(counts.values()) == 11
    assert weights["battery"] == pytest.approx(11 / 20)
    assert weights["trash"] == pytest.approx(11 / 10)


def test_weight_tensor_uses_requested_names_not_json_key_order(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(
        json.dumps({name: float(index + 1) for index, name in enumerate(reversed(CLASS_NAMES))}),
        encoding="utf-8",
    )

    tensor = ordered_weight_tensor(weights_path, CLASS_NAMES, torch.device("cpu"))

    assert tensor.dtype == torch.float32
    assert tensor.tolist() == [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
