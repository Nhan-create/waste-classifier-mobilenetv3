import json
from pathlib import Path

from src.data.schema import ManifestRecord
from src.preprocessing.eda import write_eda_report


def test_eda_json_contains_fingerprint_and_audit_counts(tmp_path: Path) -> None:
    records = [
        ManifestRecord(
            image_id="accepted",
            source_dataset="garbage_v2",
            original_label="glass",
            original_split="",
            source_path="glass/a.jpg",
            raw_path="raw/a.jpg",
            extension=".jpg",
            unified_label="glass",
            status="accepted",
            cluster_id="cluster-a",
            split="train",
        ),
        ManifestRecord(
            image_id="duplicate",
            source_dataset="vn_trash",
            original_label="PET",
            original_split="train",
            source_path="PET/b.jpg",
            raw_path="raw/b.jpg",
            extension=".jpg",
            unified_label="plastic",
            status="excluded",
            exclusion_reason="duplicate_exact",
            cluster_id="cluster-b",
            duplicate_kind="exact",
        ),
    ]

    json_path, text_path = write_eda_report(records, "f" * 64, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["dataset_fingerprint"] == "f" * 64
    assert payload["status_counts"] == {"accepted": 1, "excluded": 1}
    assert payload["exclusion_reason_counts"] == {"duplicate_exact": 1}
    assert payload["split_class_counts"] == {"train": {"glass": 1}}
    assert "Dataset fingerprint" in text_path.read_text(encoding="utf-8")
