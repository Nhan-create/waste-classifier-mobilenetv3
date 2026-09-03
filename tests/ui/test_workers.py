from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from PyQt5.QtCore import QThread

from src.inference.predict import Prediction, ScoredClass
from src.ui.workers import CameraInferenceWorker, ImageInferenceWorker


def sample_prediction() -> Prediction:
    topk = (
        ScoredClass(4, "glass", 0.62),
        ScoredClass(7, "plastic", 0.23),
        ScoredClass(5, "metal", 0.10),
    )
    return Prediction(top1=topk[0], topk=topk, low_confidence=False)


class FakePredictor:
    def __init__(self) -> None:
        self.qt_threads: list[QThread] = []
        self.received_images: list[Image.Image] = []

    def predict_pil(self, image: Image.Image, *, top_k: int = 3) -> Prediction:
        assert top_k == 3
        self.qt_threads.append(QThread.currentThread())
        self.received_images.append(image.copy())
        return sample_prediction()


class FakeCapture:
    def __init__(self, *, opened: bool = True) -> None:
        self.opened = opened
        self.released_count = 0
        self.frame = np.zeros((8, 12, 3), dtype=np.uint8)
        self.frame[..., 0] = 255

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame.copy()

    def release(self) -> None:
        self.released_count += 1


def test_static_inference_runs_on_worker_thread(qtbot, tmp_path: Path) -> None:
    predictor = FakePredictor()
    image_path = tmp_path / "glass.png"
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(image_path)
    worker = ImageInferenceWorker(predictor, image_path)

    with qtbot.waitSignal(worker.result_ready, timeout=1500) as signal:
        worker.start()
    assert worker.wait(1500)

    result = signal.args[0]
    assert result.source_path == str(image_path)
    assert result.rgb_frame.shape == (10, 20, 3)
    assert predictor.qt_threads == [worker]
    assert predictor.received_images[0].mode == "RGB"


def test_camera_worker_keeps_frames_in_memory(
    monkeypatch: pytest.MonkeyPatch,
    qtbot,
) -> None:
    capture = FakeCapture()
    predictor = FakePredictor()
    monkeypatch.setattr(
        cv2,
        "imwrite",
        lambda *args: pytest.fail("webcam wrote a temporary file"),
    )
    worker = CameraInferenceWorker(
        predictor,
        capture_factory=lambda _: capture,
        camera_index=0,
        interval_ms=20,
    )

    with qtbot.waitSignal(worker.result_ready, timeout=1500) as signal:
        worker.start()
    worker.stop()
    assert worker.wait(1500)

    result = signal.args[0]
    assert result.source_path is None
    assert result.rgb_frame[0, 0].tolist() == [0, 0, 255]
    assert predictor.received_images[0].mode == "RGB"
    assert capture.released_count == 1


def test_camera_open_failure_is_nonfatal_and_releases_capture(qtbot) -> None:
    capture = FakeCapture(opened=False)
    worker = CameraInferenceWorker(
        FakePredictor(),
        capture_factory=lambda _: capture,
        interval_ms=20,
    )

    with qtbot.waitSignal(worker.error, timeout=1500) as signal:
        worker.start()
    assert worker.wait(1500)

    assert "camera" in signal.args[0].lower()
    assert capture.released_count == 1


def test_camera_stop_is_idempotent(qtbot) -> None:
    capture = FakeCapture()
    worker = CameraInferenceWorker(
        FakePredictor(),
        capture_factory=lambda _: capture,
        interval_ms=20,
    )

    with qtbot.waitSignal(worker.result_ready, timeout=1500):
        worker.start()
    worker.stop()
    worker.stop()

    assert worker.wait(1500)
    assert capture.released_count == 1
