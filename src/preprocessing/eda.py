"""Structured exploratory reporting for manifest records."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from src.data.schema import ManifestRecord


def build_eda_summary(
    records: Sequence[ManifestRecord],
    dataset_fingerprint: str,
) -> dict[str, object]:
    status_counts = Counter(record.status for record in records)
    exclusion_counts = Counter(
        record.exclusion_reason for record in records if record.exclusion_reason
    )
    source_class: dict[str, Counter[str]] = defaultdict(Counter)
    split_class: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if record.unified_label:
            source_class[record.source_dataset][record.unified_label] += 1
        if record.split and record.status == "accepted":
            split_class[record.split][record.unified_label] += 1
    return {
        "dataset_fingerprint": dataset_fingerprint,
        "total_images": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "source_class_counts": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(source_class.items())
        },
        "split_class_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_class.items())
        },
        "duplicate_counts": {
            "exact": sum(record.duplicate_kind == "exact" for record in records),
            "near": sum(record.duplicate_kind == "near" for record in records),
            "conflict": sum(record.status == "conflict" for record in records),
        },
    }


def write_eda_report(
    records: Sequence[ManifestRecord],
    dataset_fingerprint: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    summary = build_eda_summary(records, dataset_fingerprint)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eda.json"
    text_path = output_dir / "eda_report.txt"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    text_lines = [
        "TEN-CLASS DATASET EDA",
        f"Dataset fingerprint: {dataset_fingerprint}",
        f"Total images: {summary['total_images']}",
        f"Status counts: {summary['status_counts']}",
        f"Exclusion reasons: {summary['exclusion_reason_counts']}",
        f"Duplicate counts: {summary['duplicate_counts']}",
        f"Split class counts: {summary['split_class_counts']}",
    ]
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return json_path, text_path
