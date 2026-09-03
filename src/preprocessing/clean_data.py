"""Audit a raw manifest and write deterministic duplicate reports."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.data.dedup import DuplicateReport, audit_raw_manifest
from src.data.manifest import read_manifest, write_manifest
from src.data.schema import ManifestRecord, load_label_mapping


@dataclass(frozen=True)
class CleaningOutputs:
    scanned_manifest: Path
    duplicate_report: Path
    conflict_report: Path
    report: DuplicateReport


def _atomic_write_rows(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_duplicate_reports(
    records: list[ManifestRecord],
    output_dir: Path,
) -> tuple[Path, Path]:
    cluster_counts = Counter(record.cluster_id for record in records if record.cluster_id)
    duplicate_path = output_dir / "duplicate_clusters.csv"
    duplicate_rows = [
        {
            "cluster_id": record.cluster_id,
            "image_id": record.image_id,
            "duplicate_kind": record.duplicate_kind,
            "status": record.status,
            "exclusion_reason": record.exclusion_reason,
        }
        for record in records
        if cluster_counts[record.cluster_id] > 1
    ]
    _atomic_write_rows(
        duplicate_path,
        (
            "cluster_id",
            "image_id",
            "duplicate_kind",
            "status",
            "exclusion_reason",
        ),
        duplicate_rows,
    )

    conflict_path = output_dir / "label_conflicts.csv"
    conflict_rows = [
        {
            "cluster_id": record.cluster_id,
            "image_id": record.image_id,
            "source_dataset": record.source_dataset,
            "original_label": record.original_label,
            "unified_label": record.unified_label,
        }
        for record in records
        if record.status == "conflict"
    ]
    _atomic_write_rows(
        conflict_path,
        (
            "cluster_id",
            "image_id",
            "source_dataset",
            "original_label",
            "unified_label",
        ),
        conflict_rows,
    )
    return duplicate_path, conflict_path


def run_cleaning(
    raw_manifest_path: Path,
    mapping_path: Path,
    output_dir: Path,
    *,
    phash_threshold: int,
) -> CleaningOutputs:
    """Run mapping/image audit and persist all review outputs."""

    raw_records = read_manifest(raw_manifest_path)
    mapping = load_label_mapping(mapping_path)
    audited, report = audit_raw_manifest(
        raw_records,
        mapping,
        phash_threshold=phash_threshold,
    )
    scanned_path = output_dir / "scanned_manifest.csv"
    write_manifest(scanned_path, audited)
    duplicate_path, conflict_path = _write_duplicate_reports(audited, output_dir)
    return CleaningOutputs(
        scanned_manifest=scanned_path,
        duplicate_report=duplicate_path,
        conflict_report=conflict_path,
        report=report,
    )
