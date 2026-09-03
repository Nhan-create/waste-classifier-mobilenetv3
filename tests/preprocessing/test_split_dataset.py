from dataclasses import replace
from pathlib import Path

from src.data.manifest import read_manifest, write_manifest
from src.data.schema import CLASS_NAMES, ManifestRecord
from src.data.split import SplitConfig
from src.preprocessing.split_dataset import run_split


def test_run_split_materializes_all_classes_and_records_assignments(
    tmp_path: Path,
) -> None:
    records: list[ManifestRecord] = []
    raw_root = tmp_path / "raw"
    for class_name in CLASS_NAMES:
        for index in range(3):
            image_id = f"{class_name}-{index}"
            raw_path = raw_root / f"{image_id}.jpg"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(image_id.encode("utf-8"))
            records.append(
                ManifestRecord(
                    image_id=image_id,
                    source_dataset="garbage_v2",
                    original_label=class_name,
                    original_split="",
                    source_path=f"original/{class_name}/{image_id}.jpg",
                    raw_path=str(raw_path),
                    extension=".jpg",
                    unified_label=class_name,
                    sha256=f"sha-{image_id}",
                    phash=f"{index:016x}",
                    status="accepted",
                    cluster_id=f"cluster-{image_id}",
                )
            )
    scanned = tmp_path / "scanned_manifest.csv"
    write_manifest(scanned, records)
    version_root = tmp_path / "processed" / "v1"
    split_manifest = tmp_path / "metadata" / "split_manifest.csv"

    result = run_split(
        scanned,
        version_root,
        split_manifest,
        SplitConfig(seed=42, train=0.70, val=0.15, test=0.15),
    )

    assert result == split_manifest
    assigned = read_manifest(split_manifest)
    assert {record.split for record in assigned} == {"train", "val", "test"}
    for split_name in ("train", "val", "test"):
        assert {
            path.name
            for path in (version_root / split_name).iterdir()
            if path.is_dir()
        } == set(CLASS_NAMES)
    copied = next(record for record in assigned if record.split == "train")
    assert (
        version_root
        / copied.split
        / copied.unified_label
        / f"{copied.image_id}{copied.extension}"
    ).read_bytes() == Path(copied.raw_path).read_bytes()


def test_run_split_ignores_nonaccepted_records(tmp_path: Path) -> None:
    accepted = []
    for class_name in CLASS_NAMES:
        for index in range(3):
            path = tmp_path / "raw" / f"{class_name}-{index}.jpg"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"image")
            accepted.append(
                ManifestRecord(
                    image_id=f"{class_name}-{index}",
                    source_dataset="garbage_v2",
                    original_label=class_name,
                    original_split="",
                    source_path=path.name,
                    raw_path=str(path),
                    extension=".jpg",
                    unified_label=class_name,
                    status="accepted",
                    cluster_id=f"cluster-{class_name}-{index}",
                )
            )
    excluded = replace(
        accepted[0],
        image_id="excluded",
        status="excluded",
        exclusion_reason="duplicate_exact",
    )
    manifest = tmp_path / "scanned.csv"
    write_manifest(manifest, [*accepted, excluded])

    split_manifest = tmp_path / "split.csv"
    run_split(manifest, tmp_path / "processed" / "v1", split_manifest, SplitConfig())

    assert "excluded" not in {record.image_id for record in read_manifest(split_manifest)}
