import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.schema import CLASS_NAMES
from src.evaluation.evaluate import evaluate_checkpoint
from src.models.mobilenetv3 import MODEL_ID, build_model
from src.training.checkpoint import CheckpointMetadata, save_checkpoint


def create_test_tree(root: Path) -> None:
    for index, class_name in enumerate(CLASS_NAMES):
        path = root / class_name / "one.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.random.default_rng(index).integers(
            0,
            256,
            (32, 32, 3),
            dtype=np.uint8,
        )
        Image.fromarray(pixels, mode="RGB").save(path)


def create_checkpoint(path: Path) -> None:
    model = build_model(MODEL_ID, 10, pretrained=False)
    metadata = CheckpointMetadata(
        format_version=1,
        model_name=MODEL_ID,
        num_classes=10,
        class_names=CLASS_NAMES,
        input_size=32,
        normalization={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        epoch=2,
        metrics={"val_macro_f1": 0.4},
        dataset_fingerprint="c" * 64,
        training_config={"seed": 42},
    )
    save_checkpoint(path, metadata, model.state_dict())


def test_evaluator_writes_complete_artifact_set(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"
    test_root = tmp_path / "test"
    output_dir = tmp_path / "evaluation"
    create_checkpoint(checkpoint_path)
    create_test_tree(test_root)

    result = evaluate_checkpoint(
        checkpoint_path,
        test_root,
        output_dir,
        device="cpu",
        batch_size=4,
        num_workers=0,
    )

    assert result.output_dir == output_dir
    assert {path.name for path in output_dir.iterdir()} == {
        "metrics.json",
        "per_class_metrics.csv",
        "confusion_matrix_raw.png",
        "confusion_matrix_normalized.png",
    }
    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["class_names"] == list(CLASS_NAMES)
    assert len(payload["confusion_matrix_raw"]) == 10
    assert all(len(row) == 10 for row in payload["confusion_matrix_raw"])
    with (output_dir / "per_class_metrics.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["class_id"] for row in rows] == list(CLASS_NAMES)
