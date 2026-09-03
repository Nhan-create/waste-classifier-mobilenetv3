"""Thread-safe access to the shared image predictor."""

from __future__ import annotations

import threading
from typing import Protocol

from PIL import Image

from src.inference.predict import Prediction


class Predictor(Protocol):
    """The subset of ``WastePredictor`` used by web media pipelines."""

    def predict_pil(self, image: Image.Image, *, top_k: int = 3) -> Prediction: ...


class InferenceService:
    """Serialize access to a predictor shared by Streamlit and WebRTC threads."""

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self._lock = threading.RLock()

    def predict(self, image: Image.Image, *, top_k: int = 3) -> Prediction:
        with self._lock:
            return self.predictor.predict_pil(image, top_k=top_k)
