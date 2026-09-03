"""Deterministic group-stratified dataset splitting."""

from __future__ import annotations

import hashlib
import math
import shutil
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .manifest import write_manifest
from .schema import CLASS_NAMES, ManifestRecord, PipelineError

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def ratios(self) -> dict[str, float]:
        values = {"train": self.train, "val": self.val, "test": self.test}
        if any(value <= 0 for value in values.values()):
            raise PipelineError(f"Split ratios must be positive: {values}")
        if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-9):
            raise PipelineError(f"Split ratios must sum to 1.0: {values}")
        return values


def _stable_group_key(cluster_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{cluster_id}".encode()).hexdigest()


def _allocate_bucket(
    records: Sequence[ManifestRecord],
    config: SplitConfig,
    source_dataset: str,
    unified_label: str,
) -> dict[str, list[ManifestRecord]]:
    groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if not record.cluster_id:
            raise PipelineError(
                f"Accepted record {record.image_id} has no duplicate cluster ID"
            )
        groups[record.cluster_id].append(record)
    if len(groups) < len(SPLIT_NAMES):
        raise PipelineError(
            f"Need at least 3 groups for {source_dataset} / {unified_label}; "
            f"found {len(groups)}"
        )

    ordered_groups = [
        sorted(group, key=lambda item: item.image_id)
        for _, group in sorted(
            groups.items(),
            key=lambda item: (_stable_group_key(item[0], config.seed), item[0]),
        )
    ]
    ratios = config.ratios()
    target_counts = {
        split_name: len(records) * ratios[split_name]
        for split_name in SPLIT_NAMES
    }
    assignments: dict[str, list[ManifestRecord]] = {
        split_name: [] for split_name in SPLIT_NAMES
    }
    current_counts = {split_name: 0 for split_name in SPLIT_NAMES}

    for split_name, group in zip(SPLIT_NAMES, ordered_groups[:3], strict=True):
        assignments[split_name].extend(group)
        current_counts[split_name] += len(group)

    for group in ordered_groups[3:]:
        group_size = len(group)

        def total_error(
            candidate: str,
            group_size: int = group_size,
        ) -> tuple[float, int]:
            error = 0.0
            for split_name in SPLIT_NAMES:
                count = current_counts[split_name]
                if split_name == candidate:
                    count += group_size
                error += abs(count - target_counts[split_name])
            return error, SPLIT_NAMES.index(candidate)

        destination = min(SPLIT_NAMES, key=total_error)
        assignments[destination].extend(group)
        current_counts[destination] += group_size
    return assignments


def assign_splits(
    records: Sequence[ManifestRecord],
    config: SplitConfig,
) -> dict[str, list[ManifestRecord]]:
    """Assign accepted records by source/class while preserving clusters."""

    config.ratios()
    buckets: dict[tuple[str, str], list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if record.status != "accepted":
            continue
        if record.unified_label not in CLASS_NAMES:
            raise PipelineError(
                f"Accepted record {record.image_id} has invalid class "
                f"{record.unified_label!r}"
            )
        buckets[(record.source_dataset, record.unified_label)].append(record)
    if not buckets:
        raise PipelineError("No accepted records are available for splitting")

    result: dict[str, list[ManifestRecord]] = {
        split_name: [] for split_name in SPLIT_NAMES
    }
    for (source_dataset, unified_label), bucket in sorted(buckets.items()):
        allocated = _allocate_bucket(
            bucket,
            config,
            source_dataset,
            unified_label,
        )
        for split_name in SPLIT_NAMES:
            result[split_name].extend(allocated[split_name])

    expected = set(CLASS_NAMES)
    for split_name in SPLIT_NAMES:
        result[split_name].sort(key=lambda item: item.image_id)
        actual = {record.unified_label for record in result[split_name]}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise PipelineError(
                f"Split {split_name} has invalid classes; missing={missing}, extra={extra}"
            )
    return result


def materialize_splits(
    splits: Mapping[str, Sequence[ManifestRecord]],
    version_root: Path,
    manifest_path: Path,
) -> None:
    """Copy a complete split tree without modifying an existing version."""

    if set(splits) != set(SPLIT_NAMES):
        raise PipelineError(
            f"Expected split keys {list(SPLIT_NAMES)}, received {sorted(splits)}"
        )
    if version_root.exists():
        raise PipelineError(f"Dataset version already exists: {version_root}")
    if manifest_path.exists():
        raise PipelineError(f"Split manifest already exists: {manifest_path}")

    build_root = version_root.with_name(
        f".{version_root.name}.building-{uuid.uuid4().hex}"
    )
    build_manifest = manifest_path.with_suffix(manifest_path.suffix + ".building")
    assigned_records: list[ManifestRecord] = []
    try:
        for split_name in SPLIT_NAMES:
            for class_name in CLASS_NAMES:
                (build_root / split_name / class_name).mkdir(parents=True, exist_ok=True)
            for record in sorted(splits[split_name], key=lambda item: item.image_id):
                source = Path(record.raw_path)
                if not source.is_file():
                    raise PipelineError(
                        f"Raw image for {record.image_id} not found: {source}"
                    )
                destination = (
                    build_root
                    / split_name
                    / record.unified_label
                    / f"{record.image_id}{record.extension.lower()}"
                )
                if destination.exists():
                    raise PipelineError(
                        f"Duplicate materialized image ID: {record.image_id}"
                    )
                shutil.copy2(source, destination)
                assigned_records.append(replace(record, split=split_name))

        write_manifest(build_manifest, assigned_records)
        version_root.parent.mkdir(parents=True, exist_ok=True)
        build_root.rename(version_root)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        build_manifest.replace(manifest_path)
    except Exception:
        if build_root.exists():
            shutil.rmtree(build_root)
        if build_manifest.exists():
            build_manifest.unlink()
        raise
