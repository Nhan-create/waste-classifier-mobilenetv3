import shutil
from pathlib import Path

from PIL import Image

from src.data.dedup import (
    ImageSignature,
    audit_raw_manifest,
    find_duplicate_clusters,
    hamming_distance,
)
from src.data.schema import ManifestRecord, load_label_mapping


def raw_record(path: Path, image_id: str, label: str) -> ManifestRecord:
    return ManifestRecord(
        image_id=image_id,
        source_dataset="vn_trash",
        original_label=label,
        original_split="train",
        source_path=f"Train/{label}/{path.name}",
        raw_path=str(path),
        extension=path.suffix,
    )


def test_hamming_distance_counts_different_bits() -> None:
    assert hamming_distance(0b0000, 0b1111) == 4
    assert hamming_distance(0b1010, 0b0011) == 2


def test_duplicate_clusters_include_exact_and_near_signatures() -> None:
    signatures = (
        ImageSignature(index=0, sha256="same", phash=0b0000),
        ImageSignature(index=1, sha256="same", phash=0b1111),
        ImageSignature(index=2, sha256="other", phash=0b1101),
        ImageSignature(index=3, sha256="unique", phash=0b111000),
    )

    clusters = find_duplicate_clusters(signatures, phash_threshold=1)

    assert clusters == ((0, 1, 2), (3,))


def test_exact_duplicates_keep_one_deterministic_representative(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (16, 16), "red").save(first)
    shutil.copy2(first, second)
    mapping = load_label_mapping(Path("data/metadata/label_mapping.csv"))

    audited, report = audit_raw_manifest(
        (raw_record(second, "b", "PET"), raw_record(first, "a", "PET")),
        mapping,
        phash_threshold=4,
    )

    assert [(row.image_id, row.status, row.exclusion_reason) for row in audited] == [
        ("a", "accepted", ""),
        ("b", "excluded", "duplicate_exact"),
    ]
    assert report.exact_cluster_count == 1
    assert report.near_cluster_count == 0


def test_conflicting_duplicate_labels_quarantine_the_entire_cluster(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (16, 16), "green").save(first)
    shutil.copy2(first, second)
    mapping = load_label_mapping(Path("data/metadata/label_mapping.csv"))

    audited, report = audit_raw_manifest(
        (raw_record(first, "a", "PET"), raw_record(second, "b", "Paper")),
        mapping,
        phash_threshold=4,
    )

    assert {row.status for row in audited} == {"conflict"}
    assert {row.exclusion_reason for row in audited} == {"conflicting_labels"}
    assert len(report.conflict_cluster_ids) == 1


def test_corrupt_and_unmapped_images_record_specific_reasons(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    unknown = tmp_path / "unknown.png"
    Image.new("RGB", (8, 8), "blue").save(unknown)
    mapping = load_label_mapping(Path("data/metadata/label_mapping.csv"))

    audited, _ = audit_raw_manifest(
        (
            raw_record(corrupt, "a", "PET"),
            raw_record(unknown, "b", "Mystery"),
        ),
        mapping,
        phash_threshold=4,
    )

    assert [(row.image_id, row.status, row.exclusion_reason) for row in audited] == [
        ("a", "excluded", "corrupt_image"),
        ("b", "excluded", "unmapped_label"),
    ]
