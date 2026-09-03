"""Post-materialization class and duplicate leakage validation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .dedup import ImageSignature, find_duplicate_clusters, phash_bits, sha256_file
from .manifest import read_manifest
from .schema import CLASS_NAMES, VALID_IMAGE_EXTENSIONS, ManifestRecord, PipelineError


@dataclass(frozen=True)
class ValidationReport:
    class_counts: dict[str, dict[str, int]]
    exact_cross_split_clusters: tuple[str, ...]
    near_cross_split_clusters: tuple[str, ...]
    missing_classes: dict[str, tuple[str, ...]]
    unexpected_classes: dict[str, tuple[str, ...]]
    missing_files: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not (
            self.exact_cross_split_clusters
            or self.near_cross_split_clusters
            or self.missing_classes
            or self.unexpected_classes
            or self.missing_files
        )


def _class_inventory(
    dataset_root: Path,
    expected_classes: Sequence[str],
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    expected = set(expected_classes)
    counts: dict[str, dict[str, int]] = {}
    missing: dict[str, tuple[str, ...]] = {}
    unexpected: dict[str, tuple[str, ...]] = {}
    for split_name in ("train", "val", "test"):
        split_root = dataset_root / split_name
        actual = {
            path.name for path in split_root.iterdir() if path.is_dir()
        } if split_root.is_dir() else set()
        unexpected_names = tuple(sorted(actual - expected))
        class_counts: dict[str, int] = {}
        missing_names: list[str] = []
        for class_name in expected_classes:
            class_root = split_root / class_name
            count = sum(
                1
                for path in class_root.iterdir()
                if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ) if class_root.is_dir() else 0
            class_counts[class_name] = count
            if count == 0:
                missing_names.append(class_name)
        counts[split_name] = class_counts
        if missing_names:
            missing[split_name] = tuple(missing_names)
        if unexpected_names:
            unexpected[split_name] = unexpected_names
    return counts, missing, unexpected


def _materialized_path(dataset_root: Path, record: ManifestRecord) -> Path:
    return (
        dataset_root
        / record.split
        / record.unified_label
        / f"{record.image_id}{record.extension.lower()}"
    )


def _leakage_id(records: Sequence[ManifestRecord]) -> str:
    image_ids = "\0".join(sorted(record.image_id for record in records))
    return hashlib.sha256(image_ids.encode()).hexdigest()[:24]


def validate_processed_dataset(
    dataset_root: Path,
    manifest_path: Path,
    expected_classes: Sequence[str] = CLASS_NAMES,
    phash_threshold: int = 4,
) -> ValidationReport:
    """Re-scan materialized files and report class or duplicate leakage."""

    if tuple(expected_classes) != CLASS_NAMES:
        raise PipelineError(
            f"Expected canonical class order {CLASS_NAMES}, received {tuple(expected_classes)}"
        )
    counts, missing_classes, unexpected_classes = _class_inventory(
        dataset_root,
        expected_classes,
    )
    records = [record for record in read_manifest(manifest_path) if record.split]
    signatures: list[ImageSignature] = []
    scanned_records: list[ManifestRecord] = []
    missing_files: list[str] = []
    scanned_splits: list[str] = []
    for record in records:
        path = _materialized_path(dataset_root, record)
        if not path.is_file():
            missing_files.append(str(path))
            continue
        try:
            digest = sha256_file(path)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                perceptual_hash = phash_bits(image)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise PipelineError(f"Invalid materialized image {path}: {error}") from error
        index = len(scanned_records)
        scanned_records.append(record)
        scanned_splits.append(record.split)
        signatures.append(ImageSignature(index, digest, perceptual_hash))

    exact_leaks: list[str] = []
    near_leaks: list[str] = []
    for indexes in find_duplicate_clusters(signatures, phash_threshold):
        split_names = {scanned_splits[index] for index in indexes}
        if len(split_names) < 2:
            continue
        members = [scanned_records[index] for index in indexes]
        cluster_id = _leakage_id(members)
        sha_to_splits: dict[str, set[str]] = defaultdict(set)
        for index in indexes:
            sha_to_splits[signatures[index].sha256].add(scanned_splits[index])
        if any(len(splits) > 1 for splits in sha_to_splits.values()):
            exact_leaks.append(cluster_id)
        if len(sha_to_splits) > 1:
            near_leaks.append(cluster_id)

    return ValidationReport(
        class_counts=counts,
        exact_cross_split_clusters=tuple(sorted(exact_leaks)),
        near_cross_split_clusters=tuple(sorted(near_leaks)),
        missing_classes=missing_classes,
        unexpected_classes=unexpected_classes,
        missing_files=tuple(sorted(missing_files)),
    )


def require_valid_processed_dataset(report: ValidationReport) -> None:
    if not report.is_valid:
        raise PipelineError(f"Processed dataset validation failed: {report}")
