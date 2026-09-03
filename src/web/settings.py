"""Validated runtime settings for the Streamlit application."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WebSettings:
    """All user-configurable settings used by the web frontend."""

    checkpoint_path: Path | None
    device: str = "auto"
    confidence_threshold: float = 0.55
    video_sample_fps: float = 2.0
    max_video_frames: int = 300
    live_inference_fps: float = 2.0
    smoothing_window: int = 5


def _bounded_fps(value: float, name: str) -> None:
    if not 0.0 < value <= 10.0:
        raise ValueError(f"{name} must be greater than 0 and at most 10")


def validate_settings(settings: WebSettings) -> None:
    """Reject settings that could create invalid or unbounded work."""

    if not 0.0 <= settings.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    _bounded_fps(settings.video_sample_fps, "video_sample_fps")
    _bounded_fps(settings.live_inference_fps, "live_inference_fps")
    if settings.max_video_frames <= 0:
        raise ValueError("max_video_frames must be positive")
    if settings.smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive")


def parse_web_settings(
    argv: Sequence[str],
    environ: Mapping[str, str],
) -> WebSettings:
    """Parse application arguments while ignoring Streamlit-owned arguments."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--video-sample-fps", type=float, default=2.0)
    parser.add_argument("--max-video-frames", type=int, default=300)
    parser.add_argument("--live-inference-fps", type=float, default=2.0)
    parser.add_argument("--smoothing-window", type=int, default=5)
    values, _ = parser.parse_known_args(list(argv))

    checkpoint = values.checkpoint
    if checkpoint is None and environ.get("WASTE_CHECKPOINT"):
        checkpoint = Path(environ["WASTE_CHECKPOINT"])

    settings = WebSettings(
        checkpoint_path=checkpoint,
        device=values.device,
        confidence_threshold=values.confidence_threshold,
        video_sample_fps=values.video_sample_fps,
        max_video_frames=values.max_video_frames,
        live_inference_fps=values.live_inference_fps,
        smoothing_window=values.smoothing_window,
    )
    validate_settings(settings)
    return settings


def checkpoint_problem(settings: WebSettings) -> str | None:
    """Return a user-facing checkpoint problem, or ``None`` when usable."""

    path = settings.checkpoint_path
    if path is None:
        return (
            "Chưa cấu hình checkpoint best.pt. Hãy huấn luyện mô hình trước, "
            "sau đó truyền đường dẫn bằng --checkpoint hoặc WASTE_CHECKPOINT."
        )
    if not path.is_file():
        return f"Không tìm thấy checkpoint best.pt tại: {path}"
    return None
