import json
from pathlib import Path

import pytest

from src.ui.history import HistoryRepository


def test_history_accepts_predictions_from_multiple_classes(tmp_path: Path) -> None:
    repository = HistoryRepository(tmp_path / "history.sqlite3")
    for index, class_id in enumerate(("glass", "plastic", "battery")):
        repository.add_image_prediction(
            image_path=f"image-{index}.jpg",
            model_name="mobilenet_v3_large",
            class_id=class_id,
            confidence=0.6 + index / 10,
            topk_json=json.dumps([{"class_id": class_id, "probability": 0.6}]),
            low_confidence=False,
        )

    rows = repository.list_recent(limit=10)

    assert [row.class_id for row in rows] == ["battery", "plastic", "glass"]
    assert rows[0].image_path == "image-2.jpg"
    assert rows[0].created_at


def test_history_limit_and_low_confidence_round_trip(tmp_path: Path) -> None:
    repository = HistoryRepository(tmp_path / "history.sqlite3")
    for index in range(3):
        repository.add_image_prediction(
            image_path=f"{index}.jpg",
            model_name="mobilenet_v3_large",
            class_id="trash",
            confidence=0.25,
            topk_json="[]",
            low_confidence=True,
        )

    rows = repository.list_recent(limit=2)

    assert len(rows) == 2
    assert all(row.low_confidence for row in rows)


@pytest.mark.parametrize(
    ("class_id", "confidence"),
    [("future_class", 0.5), ("glass", -0.1), ("glass", 1.1)],
)
def test_history_rejects_invalid_predictions(
    tmp_path: Path,
    class_id: str,
    confidence: float,
) -> None:
    repository = HistoryRepository(tmp_path / "history.sqlite3")

    with pytest.raises(ValueError):
        repository.add_image_prediction(
            image_path="image.jpg",
            model_name="mobilenet_v3_large",
            class_id=class_id,
            confidence=confidence,
            topk_json="[]",
            low_confidence=False,
        )


def test_history_has_no_webcam_insert_api(tmp_path: Path) -> None:
    repository = HistoryRepository(tmp_path / "history.sqlite3")

    assert not hasattr(repository, "add_webcam_prediction")
