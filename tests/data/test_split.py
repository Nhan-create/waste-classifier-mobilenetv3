from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from src.data.schema import CLASS_NAMES, ManifestRecord, PipelineError
from src.data.split import SplitConfig, assign_splits, materialize_splits


def make_records(groups_per_bucket: int = 5) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for source in ("vn_trash", "garbage_v2"):
        for class_name in CLASS_NAMES:
            for group_number in range(groups_per_bucket):
                image_id = f"{source}-{class_name}-{group_number}"
                records.append(
                    ManifestRecord(
                        image_id=image_id,
                        source_dataset=source,
                        original_label=class_name,
                        original_split="",
                        source_path=f"{class_name}/{image_id}.jpg",
                        raw_path=f"raw/{image_id}.jpg",
                        extension=".jpg",
                        unified_label=class_name,
                        sha256=f"sha-{image_id}",
                        phash=f"{group_number:016x}",
                        status="accepted",
                        cluster_id=f"cluster-{image_id}",
                    )
                )
    return records


def split_index(splits: dict[str, list[ManifestRecord]]) -> dict[str, str]:
    return {
        record.image_id: split_name
        for split_name, records in splits.items()
        for record in records
    }


def test_split_is_deterministic_and_contains_all_classes() -> None:
    records = make_records()
    config = SplitConfig(seed=42, train=0.70, val=0.15, test=0.15)

    first = assign_splits(records, config)
    second = assign_splits(list(reversed(records)), config)

    assert split_index(first) == split_index(second)
    assert {
        split_name: {record.unified_label for record in split_records}
        for split_name, split_records in first.items()
    } == {
        "train": set(CLASS_NAMES),
        "val": set(CLASS_NAMES),
        "test": set(CLASS_NAMES),
    }


def test_duplicate_cluster_never_crosses_splits() -> None:
    records = make_records()
    anchor = records[0]
    records.append(
        replace(
            anchor,
            image_id="duplicate-member",
            source_path="battery/duplicate-member.jpg",
            raw_path="raw/duplicate-member.jpg",
        )
    )

    assigned = assign_splits(records, SplitConfig())
    clusters: dict[str, set[str]] = defaultdict(set)
    for split_name, split_records in assigned.items():
        for record in split_records:
            clusters[record.cluster_id].add(split_name)

    assert all(len(split_names) == 1 for split_names in clusters.values())


def test_bucket_with_fewer_than_three_groups_is_rejected() -> None:
    records = make_records()
    records = [
        record
        for record in records
        if not (
            record.source_dataset == "vn_trash"
            and record.unified_label == "battery"
            and record.image_id.endswith(("-2", "-3", "-4"))
        )
    ]

    with pytest.raises(PipelineError, match="vn_trash.*battery"):
        assign_splits(records, SplitConfig())


def test_materialize_refuses_existing_version_without_deleting_it(
    tmp_path: Path,
) -> None:
    version_root = tmp_path / "processed" / "v1"
    version_root.mkdir(parents=True)
    sentinel = version_root / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(PipelineError, match="already exists"):
        materialize_splits(
            {"train": [], "val": [], "test": []},
            version_root,
            tmp_path / "split_manifest.csv",
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
