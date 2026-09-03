"""SQLite persistence for explicit static-image predictions only."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.data.schema import CLASS_NAMES

CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    image_path TEXT NOT NULL,
    model_name TEXT NOT NULL,
    class_id TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    topk_json TEXT NOT NULL,
    low_confidence INTEGER NOT NULL CHECK(low_confidence IN (0, 1))
)
"""


@dataclass(frozen=True)
class HistoryEntry:
    id: int
    created_at: str
    image_path: str
    model_name: str
    class_id: str
    confidence: float
    topk_json: str
    low_confidence: bool


class HistoryRepository:
    """Use short-lived connections so GUI and workers never share a connection."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(CREATE_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def add_image_prediction(
        self,
        *,
        image_path: str,
        model_name: str,
        class_id: str,
        confidence: float,
        topk_json: str,
        low_confidence: bool,
    ) -> int:
        if class_id not in CLASS_NAMES:
            raise ValueError(f"Unknown class_id: {class_id!r}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO image_predictions (
                    created_at, image_path, model_name, class_id,
                    confidence, topk_json, low_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    image_path,
                    model_name,
                    class_id,
                    confidence,
                    topk_json,
                    int(low_confidence),
                ),
            )
            return int(cursor.lastrowid)

    def list_recent(self, *, limit: int = 50) -> list[HistoryEntry]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, image_path, model_name, class_id,
                       confidence, topk_json, low_confidence
                FROM image_predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            HistoryEntry(
                id=int(row["id"]),
                created_at=str(row["created_at"]),
                image_path=str(row["image_path"]),
                model_name=str(row["model_name"]),
                class_id=str(row["class_id"]),
                confidence=float(row["confidence"]),
                topk_json=str(row["topk_json"]),
                low_confidence=bool(row["low_confidence"]),
            )
            for row in rows
        ]
