"""Bounded in-memory analysis of uploaded video files."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, BinaryIO

import av
from PIL import Image

from src.inference.predict import Prediction
from src.web.inference_service import InferenceService
from src.web.smoothing import PredictionSmoother


class VideoAnalysisError(RuntimeError):
    """An uploaded video could not be decoded or analyzed safely."""


@dataclass(frozen=True)
class SamplingPolicy:
    sample_fps: float = 2.0
    max_frames: int = 300
    fallback_fps: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 < self.sample_fps <= 10.0:
            raise ValueError("sample_fps must be greater than 0 and at most 10")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.fallback_fps <= 0.0:
            raise ValueError("fallback_fps must be positive")


class FrameSampler:
    """Accept frames at a fixed maximum frequency using media timestamps."""

    def __init__(self, policy: SamplingPolicy) -> None:
        self.policy = policy
        self._last_sample_at: float | None = None
        self._accepted = 0

    @property
    def limit_reached(self) -> bool:
        return self._accepted >= self.policy.max_frames

    def accept(self, timestamp_seconds: float) -> bool:
        if self.limit_reached:
            return False
        if not math.isfinite(timestamp_seconds):
            return False
        timestamp = max(0.0, timestamp_seconds)
        interval = 1.0 / self.policy.sample_fps
        if self._last_sample_at is not None:
            if timestamp < self._last_sample_at:
                return False
            if timestamp + 1e-9 < self._last_sample_at + interval:
                return False
        self._last_sample_at = timestamp
        self._accepted += 1
        return True


@dataclass(frozen=True)
class VideoSample:
    timestamp_seconds: float
    prediction: Prediction


@dataclass(frozen=True)
class VideoAnalysis:
    samples: tuple[VideoSample, ...]
    truncated: bool
    top1_counts: dict[str, int]


SampleCallback = Callable[[float, Image.Image, Prediction], None]


def _frame_timestamp(
    frame: Any,
    frame_index: int,
    video_stream: Any,
    fallback_fps: float,
) -> float:
    frame_time = getattr(frame, "time", None)
    if frame_time is not None:
        try:
            timestamp = float(frame_time)
            if math.isfinite(timestamp) and timestamp >= 0.0:
                return timestamp
        except (TypeError, ValueError, OverflowError):
            pass

    average_rate = getattr(video_stream, "average_rate", None)
    try:
        rate = float(average_rate) if average_rate is not None else fallback_fps
    except (TypeError, ValueError, OverflowError):
        rate = fallback_fps
    if not math.isfinite(rate) or rate <= 0.0:
        rate = fallback_fps
    return frame_index / rate


def analyze_uploaded_video(
    source: BinaryIO,
    service: InferenceService,
    policy: SamplingPolicy,
    *,
    smoothing_window: int,
    confidence_threshold: float,
    on_sample: SampleCallback | None = None,
    open_container: Callable[[BinaryIO], Any] = av.open,
) -> VideoAnalysis:
    """Decode selected frames, classify them, and retain only a light timeline."""

    container: Any | None = None
    try:
        container = open_container(source)
        video_streams = list(container.streams.video)
        if not video_streams:
            raise VideoAnalysisError("Video không chứa luồng hình ảnh có thể đọc.")
        video_stream = video_streams[0]
        sampler = FrameSampler(policy)
        smoother = PredictionSmoother(
            window_size=smoothing_window,
            confidence_threshold=confidence_threshold,
        )
        samples: list[VideoSample] = []
        counts: Counter[str] = Counter()

        for frame_index, frame in enumerate(container.decode(video=0)):
            timestamp = _frame_timestamp(
                frame,
                frame_index,
                video_stream,
                policy.fallback_fps,
            )
            if not sampler.accept(timestamp):
                if sampler.limit_reached:
                    break
                continue

            rgb_image = frame.to_image().convert("RGB")
            raw_prediction = service.predict(rgb_image, top_k=10)
            prediction = smoother.add(raw_prediction, top_k=3)
            samples.append(VideoSample(timestamp, prediction))
            counts[prediction.top1.class_id] += 1
            if on_sample is not None:
                on_sample(timestamp, rgb_image, prediction)
            if sampler.limit_reached:
                break

        if not samples:
            raise VideoAnalysisError("Không tìm thấy khung hình hợp lệ trong video.")
        return VideoAnalysis(
            samples=tuple(samples),
            truncated=sampler.limit_reached,
            top1_counts=dict(counts),
        )
    except VideoAnalysisError:
        raise
    except Exception as error:
        raise VideoAnalysisError(
            f"Không thể đọc hoặc phân tích video: {error}"
        ) from error
    finally:
        if container is not None:
            container.close()
