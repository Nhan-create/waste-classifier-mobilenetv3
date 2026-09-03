"""Throttled, thread-safe WebRTC frame classification."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import av
import cv2
import numpy as np
from PIL import Image

from src.inference.predict import Prediction
from src.web.inference_service import InferenceService
from src.web.smoothing import PredictionSmoother


def _safe_error_text(error: str | None) -> str | None:
    if error is None:
        return None
    compact = " ".join(error.split())
    return compact[:96]


def annotate_bgr_frame(
    frame: np.ndarray,
    prediction: Prediction | None,
    error: str | None,
) -> np.ndarray:
    """Draw a compact ASCII overlay supported by OpenCV's built-in font."""

    if prediction is None and error is None:
        return frame
    lines: list[str] = []
    if prediction is not None:
        lines.extend(
            f"{rank}. {row.class_id}: {row.probability:.1%}"
            for rank, row in enumerate(prediction.topk, start=1)
        )
        if prediction.low_confidence:
            lines.append("LOW CONFIDENCE - move object closer")
    if error is not None:
        lines.append(f"ERROR: {_safe_error_text(error)}")

    height, width = frame.shape[:2]
    line_height = 25
    panel_height = min(height, 14 + line_height * len(lines))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0.0, dst=frame)
    for line_index, line in enumerate(lines):
        y = 24 + line_index * line_height
        if y >= height:
            break
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


class LiveVideoCallback:
    """Classify at a bounded rate while returning every incoming frame."""

    def __init__(
        self,
        service: InferenceService,
        *,
        inference_fps: float,
        smoothing_window: int,
        confidence_threshold: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < inference_fps <= 10.0:
            raise ValueError("inference_fps must be greater than 0 and at most 10")
        self._service = service
        self._clock = clock
        self._interval = 1.0 / inference_fps
        self._state_lock = threading.RLock()
        self._next_inference_at: float | None = None
        self._latest_prediction: Prediction | None = None
        self._last_error: str | None = None
        self._smoother = PredictionSmoother(
            window_size=smoothing_window,
            confidence_threshold=confidence_threshold,
        )

    @property
    def latest_prediction(self) -> Prediction | None:
        with self._state_lock:
            return self._latest_prediction

    @property
    def last_error(self) -> str | None:
        with self._state_lock:
            return self._last_error

    def reset(self) -> None:
        with self._state_lock:
            self._next_inference_at = None
            self._latest_prediction = None
            self._last_error = None
            self._smoother.reset()

    def _claim_inference_slot(self, now: float) -> bool:
        with self._state_lock:
            if self._next_inference_at is not None and now < self._next_inference_at:
                return False
            self._next_inference_at = now + self._interval
            return True

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = np.ascontiguousarray(frame.to_ndarray(format="bgr24"))
        if self._claim_inference_slot(self._clock()):
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                raw_prediction = self._service.predict(
                    Image.fromarray(rgb),
                    top_k=10,
                )
                with self._state_lock:
                    self._latest_prediction = self._smoother.add(
                        raw_prediction,
                        top_k=3,
                    )
                    self._last_error = None
            except Exception as error:  # noqa: BLE001 - a media callback must return.
                with self._state_lock:
                    self._last_error = str(error)

        with self._state_lock:
            prediction = self._latest_prediction
            error_text = self._last_error
        annotate_bgr_frame(bgr, prediction, error_text)
        return av.VideoFrame.from_ndarray(bgr, format="bgr24")
