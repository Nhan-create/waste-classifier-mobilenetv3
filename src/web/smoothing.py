"""Temporal probability smoothing shared by uploaded and live video."""

from __future__ import annotations

from collections import deque

from src.inference.predict import Prediction, ScoredClass


class PredictionSmoother:
    """Average complete class-probability vectors over a bounded window."""

    def __init__(self, *, window_size: int, confidence_threshold: float) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self._vectors: deque[dict[str, tuple[int, float]]] = deque(
            maxlen=window_size
        )
        self._class_map: dict[str, int] | None = None

    @property
    def sample_count(self) -> int:
        return len(self._vectors)

    def reset(self) -> None:
        self._vectors.clear()
        self._class_map = None

    def add(self, prediction: Prediction, *, top_k: int = 3) -> Prediction:
        vector: dict[str, tuple[int, float]] = {}
        seen_indices: set[int] = set()
        for scored in prediction.topk:
            if scored.class_id in vector or scored.index in seen_indices:
                raise ValueError("prediction contains duplicate class IDs or indices")
            if not 0.0 <= scored.probability <= 1.0:
                raise ValueError("prediction probabilities must be between 0 and 1")
            vector[scored.class_id] = (scored.index, scored.probability)
            seen_indices.add(scored.index)

        if not vector:
            raise ValueError("prediction must contain at least one class")
        class_map = {class_id: index for class_id, (index, _) in vector.items()}
        if self._class_map is None:
            self._class_map = class_map
        elif class_map != self._class_map:
            raise ValueError("prediction class/index mapping changed between frames")
        if not 1 <= top_k <= len(vector):
            raise ValueError("top_k must be within the available class count")

        self._vectors.append(vector)
        averaged = []
        for class_id, index in self._class_map.items():
            probability = sum(
                sample[class_id][1] for sample in self._vectors
            ) / len(self._vectors)
            averaged.append(
                ScoredClass(
                    index=index,
                    class_id=class_id,
                    probability=probability,
                )
            )
        averaged.sort(key=lambda row: (-row.probability, row.index))
        ranked = tuple(averaged[:top_k])
        return Prediction(
            top1=ranked[0],
            topk=ranked,
            low_confidence=ranked[0].probability < self.confidence_threshold,
        )
