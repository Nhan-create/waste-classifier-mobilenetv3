from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.data.schema import CLASS_NAMES
from src.models.mobilenetv3 import MODEL_ID, build_model
from src.training.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointMetadata,
    build_model_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        format_version=1,
        model_name=MODEL_ID,
        num_classes=10,
        class_names=CLASS_NAMES,
        input_size=224,
        normalization={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        epoch=4,
        metrics={"val_macro_f1": 0.63},
        dataset_fingerprint="a" * 64,
        training_config={"seed": 42},
    )


def test_best_and_last_use_one_envelope(tmp_path: Path) -> None:
    model = build_model(MODEL_ID, 10, pretrained=False)
    best_path = tmp_path / "best.pt"
    last_path = tmp_path / "last.pt"

    save_checkpoint(best_path, metadata(), model.state_dict())
    save_checkpoint(
        last_path,
        metadata(),
        model.state_dict(),
        resume_state={"optimizer": {"state": {}}, "next_epoch": 5},
    )

    best = load_checkpoint(best_path, torch.device("cpu"))
    last = load_checkpoint(last_path, torch.device("cpu"))
    assert best.metadata == metadata()
    assert best.resume_state is None
    assert last.resume_state == {"optimizer": {"state": {}}, "next_epoch": 5}


def test_reloaded_model_produces_identical_logits(tmp_path: Path) -> None:
    torch.manual_seed(7)
    source = build_model(MODEL_ID, 10, pretrained=False).eval()
    input_tensor = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        expected = source(input_tensor)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, metadata(), source.state_dict())

    loaded = load_checkpoint(checkpoint_path, torch.device("cpu"))
    restored = build_model_from_checkpoint(loaded, torch.device("cpu"))
    with torch.inference_mode():
        actual = restored(input_tensor)

    torch.testing.assert_close(actual, expected)


def test_expected_class_order_mismatch_is_rejected(tmp_path: Path) -> None:
    model = build_model(MODEL_ID, 10, pretrained=False)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, metadata(), model.state_dict())

    with pytest.raises(CheckpointCompatibilityError, match="class order"):
        load_checkpoint(
            checkpoint_path,
            torch.device("cpu"),
            expected_classes=tuple(reversed(CLASS_NAMES)),
        )


def test_unknown_format_and_wrong_classifier_head_are_rejected(
    tmp_path: Path,
) -> None:
    model = build_model(MODEL_ID, 10, pretrained=False)
    valid_path = tmp_path / "valid.pt"
    save_checkpoint(valid_path, metadata(), model.state_dict())
    envelope = torch.load(valid_path, map_location="cpu", weights_only=False)

    bad_format_path = tmp_path / "bad-format.pt"
    envelope["metadata"]["format_version"] = 99
    torch.save(envelope, bad_format_path)
    with pytest.raises(CheckpointCompatibilityError, match="format_version"):
        load_checkpoint(bad_format_path, torch.device("cpu"))

    bad_head_path = tmp_path / "bad-head.pt"
    envelope["metadata"]["format_version"] = 1
    envelope["model_state"]["classifier.3.weight"] = torch.zeros(
        9,
        model.classifier[3].in_features,
    )
    torch.save(envelope, bad_head_path)
    with pytest.raises(CheckpointCompatibilityError, match="classifier"):
        load_checkpoint(bad_head_path, torch.device("cpu"))


def test_invalid_dataset_fingerprint_is_rejected(tmp_path: Path) -> None:
    model = build_model(MODEL_ID, 10, pretrained=False)

    with pytest.raises(CheckpointCompatibilityError, match="fingerprint"):
        save_checkpoint(
            tmp_path / "bad.pt",
            replace(metadata(), dataset_fingerprint="not-sha256"),
            model.state_dict(),
        )
