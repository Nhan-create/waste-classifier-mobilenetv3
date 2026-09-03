"""Desktop entry point for the ten-class waste classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from src.inference.predict import WastePredictor
from src.ui.history import HistoryRepository
from src.ui.main_window import WasteClassifierWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify ten waste categories with MobileNetV3-Large"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--history-db",
        type=Path,
        default=Path("outputs/history.sqlite3"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not arguments.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {arguments.checkpoint}")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    application = QApplication.instance() or QApplication(sys.argv)
    predictor = WastePredictor(
        arguments.checkpoint,
        device=arguments.device,
        confidence_threshold=arguments.confidence_threshold,
    )
    history = HistoryRepository(arguments.history_db)
    window = WasteClassifierWindow(
        predictor,
        history,
        camera_index=arguments.camera_index,
    )
    window.show()
    return application.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
