"""Qt workers that keep model inference and camera I/O off the UI thread."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

from src.inference.predict import Prediction


class Predictor(Protocol):
    def predict_pil(self, image: Image.Image, *, top_k: int = 3) -> Prediction: ...


class Capture(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray]: ...

    def release(self) -> None: ...


@dataclass(frozen=True)
class FramePrediction:
    rgb_frame: np.ndarray
    prediction: Prediction
    source_path: str | None


class ImageInferenceWorker(QThread):
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, predictor: Predictor, image_path: Path | str) -> None:
        super().__init__()
        self.predictor = predictor
        self.image_path = Path(image_path)

    def run(self) -> None:
        try:
            with Image.open(self.image_path) as image:
                image.load()
                rgb = image.convert("RGB")
                prediction = self.predictor.predict_pil(rgb, top_k=3)
                self.result_ready.emit(
                    FramePrediction(
                        rgb_frame=np.asarray(rgb).copy(),
                        prediction=prediction,
                        source_path=str(self.image_path),
                    )
                )
        except Exception as error:  # noqa: BLE001 - convert worker failures to Qt signal
            self.error.emit(f"Không thể phân loại {self.image_path}: {error}")


class CameraInferenceWorker(QThread):
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        predictor: Predictor,
        *,
        capture_factory: Callable[[int], Capture] = cv2.VideoCapture,
        camera_index: int = 0,
        interval_ms: int = 250,
    ) -> None:
        super().__init__()
        if interval_ms < 1:
            raise ValueError("interval_ms must be at least 1")
        self.predictor = predictor
        self.capture_factory = capture_factory
        self.camera_index = camera_index
        self.interval_ms = interval_ms
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        capture: Capture | None = None
        try:
            capture = self.capture_factory(self.camera_index)
            if not capture.isOpened():
                self.error.emit(f"Không mở được camera {self.camera_index}.")
                return
            while not self._stop_event.is_set():
                started_at = time.monotonic()
                success, bgr_frame = capture.read()
                if not success:
                    self.error.emit("Không đọc được khung hình từ camera.")
                    return
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                prediction = self.predictor.predict_pil(
                    Image.fromarray(rgb_frame),
                    top_k=3,
                )
                self.result_ready.emit(
                    FramePrediction(
                        rgb_frame=rgb_frame.copy(),
                        prediction=prediction,
                        source_path=None,
                    )
                )
                elapsed = time.monotonic() - started_at
                remaining = max(0.0, self.interval_ms / 1000 - elapsed)
                self._stop_event.wait(remaining)
        except Exception as error:  # noqa: BLE001 - convert worker failures to Qt signal
            self.error.emit(f"Lỗi xử lý camera: {error}")
        finally:
            if capture is not None:
                capture.release()
