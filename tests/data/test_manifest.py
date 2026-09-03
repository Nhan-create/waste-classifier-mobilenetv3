from dataclasses import replace
from pathlib import Path

from src.data.manifest import fingerprint_manifest, read_manifest, write_manifest
from src.data.schema import ManifestRecord


def make_record(image_id: str) -> ManifestRecord:
    return ManifestRecord(
        image_id=image_id,
        source_dataset="vn_trash",
        original_label="PET",
        original_split="train",
        source_path=f"Train/PET/{image_id}.jpg",
        raw_path=f"data/raw/vn_trash/PET/{image_id}.jpg",
        extension=".jpg",
    )


def test_manifest_round_trip_is_sorted_and_lossless(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    records = [make_record("b"), make_record("a")]

    write_manifest(path, records)

    assert read_manifest(path) == [make_record("a"), make_record("b")]
    assert path.read_text(encoding="utf-8").splitlines()[1].startswith("a,")


def test_fingerprint_ignores_record_order_and_tracks_inputs() -> None:
    first_record = make_record("1")
    second_record = replace(
        first_record,
        image_id="2",
        source_path="Train/PET/2.jpg",
        raw_path="data/raw/vn_trash/PET/2.jpg",
    )

    baseline = fingerprint_manifest(
        [first_record, second_record],
        b"mapping-a",
        b'{"seed":42}',
    )

    assert baseline == fingerprint_manifest(
        [second_record, first_record],
        b"mapping-a",
        b'{"seed":42}',
    )
    assert baseline != fingerprint_manifest(
        [first_record, second_record],
        b"mapping-b",
        b'{"seed":42}',
    )
    assert baseline != fingerprint_manifest(
        [first_record, second_record],
        b"mapping-a",
        b'{"seed":43}',
    )
