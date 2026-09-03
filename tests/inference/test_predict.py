import re
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from src.data.schema import CLASS_NAMES
from src.inference.predict import PredictionError, WastePredictor
from src.models.mobilenetv3 import MODEL_ID
from src.training.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointMetadata,
    LoadedCheckpoint,
)


def loaded_checkpoint() -> LoadedCheckpoint:
    metadata = CheckpointMetadata(
        format_version=1,
        model_name=MODEL_ID,
        num_classes=10,
        class_names=CLASS_NAMES,
        input_size=32,
        normalization={"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        epoch=1,
        metrics={"val_macro_f1": 0.5},
        dataset_fingerprint="d" * 64,
        training_config={"seed": 42},
    )
    return LoadedCheckpoint(metadata=metadata, model_state={}, resume_state=None)


class FixedLogitModel(nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))
        self.received_shapes: list[tuple[int, ...]] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.received_shapes.append(tuple(inputs.shape))
        return self.fixed_logits.unsqueeze(0).expand(inputs.shape[0], -1)


def test_predictor_does_not_load_checkpoint_until_first_prediction(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.pt"
    predictor = WastePredictor(missing, device="cpu")

    assert not predictor.is_loaded
    with pytest.raises(CheckpointCompatibilityError, match=re.escape(str(missing))):
        predictor.predict_pil(Image.new("RGB", (10, 10)), top_k=3)


def test_top_three_indices_are_mapped_through_checkpoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FixedLogitModel([0.0, 0.1, 0.2, 0.3, 4.0, 3.0, 0.4, 5.0, 0.5, 0.6])
    monkeypatch.setattr(
        "src.inference.predict.load_checkpoint",
        lambda path, device: loaded_checkpoint(),
    )
    monkeypatch.setattr(
        "src.inference.predict.build_model_from_checkpoint",
        lambda checkpoint, device: model,
    )
    predictor = WastePredictor(Path("unused.pt"), device="cpu", confidence_threshold=0.55)

    result = predictor.predict_pil(Image.new("L", (20, 25), color=120), top_k=3)

    assert [item.class_id for item in result.topk] == ["plastic", "glass", "metal"]
    assert result.top1 == result.topk[0]
    assert not result.low_confidence
    assert model.received_shapes == [(1, 3, 32, 32)]


def test_low_confidence_keeps_the_top_one_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FixedLogitModel([0.0] * 10)
    monkeypatch.setattr(
        "src.inference.predict.load_checkpoint",
        lambda path, device: loaded_checkpoint(),
    )
    monkeypatch.setattr(
        "src.inference.predict.build_model_from_checkpoint",
        lambda checkpoint, device: model,
    )
    predictor = WastePredictor(Path("unused.pt"), device="cpu", confidence_threshold=0.55)

    result = predictor.predict_pil(Image.new("RGB", (16, 16)), top_k=3)

    assert result.low_confidence
    assert result.top1 == result.topk[0]
    assert result.top1.probability == pytest.approx(0.1)


def test_corrupt_image_error_names_the_file(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    predictor = WastePredictor(tmp_path / "unused.pt", device="cpu")

    with pytest.raises(PredictionError, match=re.escape(str(corrupt))):
        predictor.predict_path(corrupt)
