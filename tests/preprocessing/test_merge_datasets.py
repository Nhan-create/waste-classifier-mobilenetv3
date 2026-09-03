import re
from pathlib import Path

import pytest
from PIL import Image

from src.data.manifest import read_manifest
from src.data.schema import PipelineError
from src.preprocessing.merge_datasets import SourceSpec, merge_sources


def save_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def test_merge_preserves_provenance_and_avoids_name_collisions(tmp_path: Path) -> None:
    vn_root = tmp_path / "vn"
    garbage_root = tmp_path / "garbage"
    vn_image = vn_root / "Train" / "PET" / "same.jpg"
    garbage_image = garbage_root / "original" / "plastic" / "same.jpg"
    save_image(vn_image, "red")
    (vn_root / "Test").mkdir()
    save_image(garbage_image, "blue")
    manifest_path = tmp_path / "metadata" / "raw_manifest.csv"

    records = merge_sources(
        (
            SourceSpec("vn_trash", vn_root, "vn_trash"),
            SourceSpec("garbage_v2", garbage_root, "garbage_v2"),
        ),
        tmp_path / "raw",
        manifest_path,
    )

    assert {record.source_dataset for record in records} == {
        "vn_trash",
        "garbage_v2",
    }
    assert {record.original_split for record in records} == {"train", ""}
    assert len({record.image_id for record in records}) == 2
    assert len({record.raw_path for record in records}) == 2
    assert {Path(record.raw_path).read_bytes() for record in records} == {
        vn_image.read_bytes(),
        garbage_image.read_bytes(),
    }
    assert read_manifest(manifest_path) == records


def test_merge_rerun_produces_identical_manifest(tmp_path: Path) -> None:
    vn_root = tmp_path / "vn"
    save_image(vn_root / "Train" / "Paper" / "one.png", "white")
    save_image(vn_root / "Test" / "Paper" / "two.png", "gray")
    manifest_path = tmp_path / "metadata" / "raw_manifest.csv"
    sources = (SourceSpec("vn_trash", vn_root, "vn_trash"),)

    first_records = merge_sources(sources, tmp_path / "raw", manifest_path)
    first_bytes = manifest_path.read_bytes()
    second_records = merge_sources(sources, tmp_path / "raw", manifest_path)

    assert second_records == first_records
    assert manifest_path.read_bytes() == first_bytes


def test_missing_source_names_the_expected_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing-vn"

    with pytest.raises(PipelineError, match=re.escape(str(missing))):
        merge_sources(
            (SourceSpec("vn_trash", missing, "vn_trash"),),
            tmp_path / "raw",
            tmp_path / "manifest.csv",
        )


def test_vn_layout_requires_train_and_test_directories(tmp_path: Path) -> None:
    vn_root = tmp_path / "vn"
    save_image(vn_root / "Train" / "PET" / "one.jpg", "green")

    with pytest.raises(PipelineError, match="Test"):
        merge_sources(
            (SourceSpec("vn_trash", vn_root, "vn_trash"),),
            tmp_path / "raw",
            tmp_path / "manifest.csv",
        )


def test_unsupported_files_are_not_added_to_manifest(tmp_path: Path) -> None:
    garbage_root = tmp_path / "garbage" / "original" / "glass"
    garbage_root.mkdir(parents=True)
    (garbage_root / "animation.gif").write_bytes(b"GIF89a")

    records = merge_sources(
        (SourceSpec("garbage_v2", tmp_path / "garbage", "garbage_v2"),),
        tmp_path / "raw",
        tmp_path / "manifest.csv",
    )

    assert records == []
