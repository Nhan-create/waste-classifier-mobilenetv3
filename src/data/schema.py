"""Canonical labels and manifest schema for the ten-class dataset."""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES: tuple[str, ...] = (
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
)

VALID_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


class PipelineError(RuntimeError):
    """A deterministic dataset build cannot continue."""


class LabelMappingError(PipelineError):
    """A source label has no valid canonical mapping."""


def canonicalize_label(value: str) -> str:
    """Normalize source labels without inventing semantic mappings."""

    normalized = re.sub(r"[-\s]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", normalized)


def load_label_mapping(path: Path) -> dict[tuple[str, str], str]:
    """Load and validate the committed source-to-canonical mapping."""

    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            expected_header = ["source_dataset", "original_label", "unified_label"]
            if reader.fieldnames != expected_header:
                raise LabelMappingError(
                    f"Invalid label mapping header in {path}; expected {expected_header}"
                )
            rows = list(reader)
    except FileNotFoundError as error:
        raise LabelMappingError(f"Label mapping not found: {path}") from error

    if not rows:
        raise LabelMappingError(f"Label mapping is empty: {path}")

    mapping: dict[tuple[str, str], str] = {}
    for row in rows:
        source = canonicalize_label(row["source_dataset"])
        original = canonicalize_label(row["original_label"])
        unified = canonicalize_label(row["unified_label"])
        key = (source, original)
        if not source or not original:
            raise LabelMappingError(f"Blank source or original label in {path}: {row}")
        if unified not in CLASS_NAMES:
            raise LabelMappingError(f"Unknown canonical label {unified!r} in {path}")
        if key in mapping:
            raise LabelMappingError(f"Duplicate mapping for {key!r} in {path}")
        mapping[key] = unified
    return mapping


def map_to_unified_label(
    source_dataset: str,
    original_label: str,
    mapping: Mapping[tuple[str, str], str],
) -> str:
    """Map one source label or fail instead of guessing a class."""

    source = canonicalize_label(source_dataset)
    original = canonicalize_label(original_label)
    if source == "garbage_v2" and original in CLASS_NAMES:
        return original
    try:
        return mapping[(source, original)]
    except KeyError as error:
        raise LabelMappingError(
            f"Unmapped label {original_label!r} for source {source_dataset!r}"
        ) from error


@dataclass(frozen=True)
class ManifestRecord:
    """One image and all provenance/audit fields used by the pipeline."""

    image_id: str
    source_dataset: str
    original_label: str
    original_split: str
    source_path: str
    raw_path: str
    extension: str
    unified_label: str = ""
    sha256: str = ""
    phash: str = ""
    width: int = 0
    height: int = 0
    mode: str = ""
    status: str = "pending"
    exclusion_reason: str = ""
    cluster_id: str = ""
    duplicate_kind: str = ""
    split: str = ""
