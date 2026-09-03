import csv
import shutil
from pathlib import Path

from PIL import Image

from src.data.manifest import write_manifest
from src.data.schema import ManifestRecord
from src.preprocessing.clean_data import run_cleaning


def make_record(path: Path, image_id: str, label: str) -> ManifestRecord:
    return ManifestRecord(
        image_id=image_id,
        source_dataset="vn_trash",
        original_label=label,
        original_split="train",
        source_path=f"Train/{label}/{path.name}",
        raw_path=str(path),
        extension=path.suffix,
    )


def test_cleaning_writes_scanned_duplicate_and_conflict_reports(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (16, 16), "yellow").save(first)
    shutil.copy2(first, second)
    raw_manifest = tmp_path / "raw_manifest.csv"
    write_manifest(
        raw_manifest,
        (make_record(first, "a", "PET"), make_record(second, "b", "Paper")),
    )
    output_dir = tmp_path / "metadata"

    result = run_cleaning(
        raw_manifest,
        Path("data/metadata/label_mapping.csv"),
        output_dir,
        phash_threshold=4,
    )

    assert result.scanned_manifest == output_dir / "scanned_manifest.csv"
    assert result.duplicate_report == output_dir / "duplicate_clusters.csv"
    assert result.conflict_report == output_dir / "label_conflicts.csv"
    with result.conflict_report.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["image_id"] for row in rows} == {"a", "b"}
    assert {row["unified_label"] for row in rows} == {"paper", "plastic"}
