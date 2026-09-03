"""Framework-independent helpers for Streamlit media rendering."""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from src.inference.predict import Prediction
from src.ui.presentation import present_label
from src.web.inference_service import InferenceService


class MediaInputError(RuntimeError):
    """Browser-provided media is not a readable image."""


@dataclass(frozen=True)
class ScoredClassView:
    class_id: str
    display_name: str
    icon: str
    color: str
    probability: float


@dataclass(frozen=True)
class PredictionView:
    class_id: str
    display_name: str
    icon: str
    color: str
    probability: float
    topk: tuple[ScoredClassView, ...]
    low_confidence: bool


def classify_image_bytes(
    payload: bytes,
    service: InferenceService,
) -> tuple[Image.Image, Prediction]:
    """Decode one image entirely in memory and classify it as RGB."""

    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            image = opened.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise MediaInputError(
            "Không thể đọc ảnh. Hãy dùng JPEG, PNG, WebP hoặc BMP hợp lệ."
        ) from error
    return image, service.predict(image, top_k=3)


def prediction_view(prediction: Prediction) -> PredictionView:
    """Convert stable class IDs to Vietnamese display metadata."""

    rows = []
    for scored in prediction.topk:
        label = present_label(scored.class_id)
        rows.append(
            ScoredClassView(
                class_id=scored.class_id,
                display_name=label.display_name,
                icon=label.icon,
                color=label.color,
                probability=scored.probability,
            )
        )
    top = present_label(prediction.top1.class_id)
    return PredictionView(
        class_id=prediction.top1.class_id,
        display_name=top.display_name,
        icon=top.icon,
        color=top.color,
        probability=prediction.top1.probability,
        topk=tuple(rows),
        low_confidence=prediction.low_confidence,
    )
