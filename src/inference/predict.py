"""Metadata-safe, lazy image prediction with MobileNetV3."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import transforms

from src.training.checkpoint import (
    LoadedCheckpoint,
    build_model_from_checkpoint,
    load_checkpoint,
)


class PredictionError(RuntimeError):
    """An input image or inference setting cannot be used safely."""


@dataclass(frozen=True)
class ScoredClass:
    index: int
    class_id: str
    probability: float


@dataclass(frozen=True)
class Prediction:
    top1: ScoredClass
    topk: tuple[ScoredClass, ...]
    low_confidence: bool


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized not in {"cpu", "cuda"}:
        raise PredictionError("device must be one of: auto, cpu, cuda")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise PredictionError("CUDA was requested but is not available")
    return torch.device(normalized)


class WastePredictor:
    """Load a checkpoint on demand and reuse it for subsequent predictions."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        *,
        device: str = "auto",
        confidence_threshold: float = 0.55,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.checkpoint_path = Path(checkpoint_path)
        self.device = _resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self._checkpoint: LoadedCheckpoint | None = None
        self._model: nn.Module | None = None
        self._transform: transforms.Compose | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        metadata = checkpoint.metadata
        self._model = build_model_from_checkpoint(checkpoint, self.device)
        self._transform = transforms.Compose(
            [
                transforms.Resize((metadata.input_size, metadata.input_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=metadata.normalization["mean"],
                    std=metadata.normalization["std"],
                ),
            ]
        )
        self._checkpoint = checkpoint

    def predict_pil(self, image: Image.Image, *, top_k: int = 3) -> Prediction:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._ensure_loaded()
        assert self._checkpoint is not None
        assert self._model is not None
        assert self._transform is not None

        class_names = self._checkpoint.metadata.class_names
        if top_k > len(class_names):
            raise ValueError(f"top_k cannot exceed {len(class_names)}")
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self._model(tensor), dim=1)[0]
        values, indices = probabilities.topk(top_k)
        ranked = tuple(
            ScoredClass(
                index=int(index),
                class_id=class_names[int(index)],
                probability=float(value),
            )
            for value, index in zip(values.cpu(), indices.cpu(), strict=True)
        )
        return Prediction(
            top1=ranked[0],
            topk=ranked,
            low_confidence=ranked[0].probability < self.confidence_threshold,
        )

    def predict_path(self, image_path: Path | str, *, top_k: int = 3) -> Prediction:
        path = Path(image_path)
        try:
            with Image.open(path) as image:
                image.load()
                return self.predict_pil(image, top_k=top_k)
        except (FileNotFoundError, OSError, UnidentifiedImageError) as error:
            raise PredictionError(f"Cannot read image {path}: {error}") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify one waste image")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        predictor = WastePredictor(
            args.checkpoint,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
        )
        result = predictor.predict_path(args.image, top_k=args.top_k)
    except (PredictionError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
