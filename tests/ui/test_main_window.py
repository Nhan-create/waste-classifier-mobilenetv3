from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.inference.predict import Prediction, ScoredClass
from src.ui.history import HistoryRepository
from src.ui.main_window import WasteClassifierWindow
from src.ui.workers import FramePrediction


class UnusedPredictor:
    def predict_pil(self, image, *, top_k=3):
        raise AssertionError("direct result rendering should not run inference")


def prediction(*, low_confidence: bool = False) -> Prediction:
    topk = (
        ScoredClass(4, "glass", 0.62),
        ScoredClass(7, "plastic", 0.23),
        ScoredClass(5, "metal", 0.10),
    )
    return Prediction(top1=topk[0], topk=topk, low_confidence=low_confidence)


def rgb_frame() -> np.ndarray:
    return np.full((24, 32, 3), (40, 120, 200), dtype=np.uint8)


@pytest.fixture
def history(tmp_path: Path) -> HistoryRepository:
    return HistoryRepository(tmp_path / "history.sqlite3")


@pytest.fixture
def window(qtbot, history: HistoryRepository) -> WasteClassifierWindow:
    widget = WasteClassifierWindow(UnusedPredictor(), history)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def test_window_renders_top_one_top_three_and_warning(
    qtbot,
    window: WasteClassifierWindow,
) -> None:
    result = FramePrediction(rgb_frame(), prediction(), "glass.jpg")

    window.handle_result(result)

    assert "Thủy tinh" in window.result_label.text()
    assert "62.0%" in window.confidence_label.text()
    assert window.top_three.count() == 3
    assert window.warning_label.isHidden()

    window.handle_result(replace(result, prediction=prediction(low_confidence=True)))
    qtbot.wait(10)

    assert window.warning_label.isVisible()
    assert "Thủy tinh" in window.result_label.text()


def test_only_static_image_results_are_persisted(
    window: WasteClassifierWindow,
    history: HistoryRepository,
) -> None:
    window.handle_result(FramePrediction(rgb_frame(), prediction(), "glass.jpg"))
    window.handle_result(FramePrediction(rgb_frame(), prediction(), None))

    rows = history.list_recent(limit=10)

    assert len(rows) == 1
    assert rows[0].image_path == "glass.jpg"
    assert rows[0].class_id == "glass"
    assert '"plastic"' in rows[0].topk_json


def test_camera_error_does_not_disable_image_button(
    window: WasteClassifierWindow,
) -> None:
    window.handle_camera_error("Không mở được camera")

    assert window.camera_button.isEnabled()
    assert window.select_image_button.isEnabled()
    assert "Không mở được camera" in window.status_label.text()


def test_webcam_result_does_not_change_camera_button_state(
    window: WasteClassifierWindow,
) -> None:
    window.camera_button.setText("Tắt camera")

    window.handle_result(FramePrediction(rgb_frame(), prediction(), None))

    assert window.camera_button.text() == "Tắt camera"
