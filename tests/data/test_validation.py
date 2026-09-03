from pathlib import Path

import numpy as np
from PIL import Image

from src.data.manifest import write_manifest
from src.data.schema import CLASS_NAMES, ManifestRecord
from src.data.validation import validate_processed_dataset


def save_noise(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def build_valid_tree(root: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    sequence = 0
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            image_id = f"{split}-{class_name}"
            path = root / split / class_name / f"{image_id}.png"
            save_noise(path, sequence)
            sequence += 1
            records.append(
                ManifestRecord(
                    image_id=image_id,
                    source_dataset="garbage_v2",
                    original_label=class_name,
                    original_split="",
                    source_path=f"original/{class_name}/{image_id}.png",
                    raw_path="unused",
                    extension=".png",
                    unified_label=class_name,
                    status="accepted",
                    cluster_id=f"source-{image_id}",
                    split=split,
                )
            )
    return records


def test_validator_accepts_complete_leakage_free_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "v1"
    records = build_valid_tree(dataset_root)
    manifest = tmp_path / "split_manifest.csv"
    write_manifest(manifest, records)

    report = validate_processed_dataset(dataset_root, manifest, CLASS_NAMES, 4)

    assert report.is_valid
    assert report.exact_cross_split_clusters == ()
    assert report.near_cross_split_clusters == ()
    assert report.missing_classes == {}


def test_validator_reports_exact_and_near_cross_split_leakage(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "v1"
    records = build_valid_tree(dataset_root)
    train_battery = dataset_root / "train" / "battery" / "train-battery.png"
    val_battery = dataset_root / "val" / "battery" / "val-battery.png"
    val_battery.write_bytes(train_battery.read_bytes())

    train_glass = dataset_root / "train" / "glass" / "train-glass.png"
    val_glass = dataset_root / "val" / "glass" / "val-glass.png"
    pixels = np.asarray(Image.open(train_glass)).copy()
    pixels[0, 0, 0] ^= 1
    Image.fromarray(pixels, mode="RGB").save(val_glass)
    manifest = tmp_path / "split_manifest.csv"
    write_manifest(manifest, records)

    report = validate_processed_dataset(dataset_root, manifest, CLASS_NAMES, 4)

    assert len(report.exact_cross_split_clusters) == 1
    assert len(report.near_cross_split_clusters) == 1
    assert not report.is_valid


def test_validator_reports_missing_class_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "v1"
    records = build_valid_tree(dataset_root)
    missing_file = dataset_root / "test" / "trash" / "test-trash.png"
    missing_file.unlink()
    missing_file.parent.rmdir()
    manifest = tmp_path / "split_manifest.csv"
    write_manifest(manifest, [row for row in records if row.image_id != "test-trash"])

    report = validate_processed_dataset(dataset_root, manifest, CLASS_NAMES, 4)

    assert report.missing_classes == {"test": ("trash",)}
    assert not report.is_valid
