"""End-to-end reproducible ten-class dataset build."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from src.data.manifest import fingerprint_manifest, read_manifest
from src.data.schema import PipelineError
from src.data.split import SplitConfig
from src.data.validation import (
    ValidationReport,
    require_valid_processed_dataset,
    validate_processed_dataset,
)
from src.preprocessing.balance_classes import write_class_weights
from src.preprocessing.clean_data import run_cleaning
from src.preprocessing.eda import write_eda_report
from src.preprocessing.merge_datasets import SourceSpec, merge_sources
from src.preprocessing.split_dataset import run_split

_PINNED_VERSIONS = {
    "vn_trash": ("mrgetshjtdone/vn-trash-classification", 1),
    "garbage_v2": ("sumn2u/garbage-classification-v2", 12),
}


@dataclass(frozen=True)
class PipelineResult:
    processed_root: Path
    split_manifest: Path
    fingerprint: str
    validation: ValidationReport
    class_weights: Path
    eda_json: Path
    eda_text: Path


def _load_config(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as error:
        raise PipelineError(f"Pipeline config not found: {path}") from error
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise PipelineError(f"Config has no dataset mapping: {path}")
    return config


def _source_specs(config: dict) -> tuple[SourceSpec, ...]:
    source_config = config["dataset"].get("sources")
    if not isinstance(source_config, dict) or not source_config:
        raise PipelineError("Config must define at least one dataset source")
    specs: list[SourceSpec] = []
    for source_name, values in source_config.items():
        if source_name not in _PINNED_VERSIONS:
            raise PipelineError(f"Unsupported dataset source: {source_name}")
        if not isinstance(values, dict) or "root" not in values:
            raise PipelineError(f"Dataset source {source_name} has no local root")
        expected_slug, expected_version = _PINNED_VERSIONS[source_name]
        if (values.get("slug"), values.get("version")) != (
            expected_slug,
            expected_version,
        ):
            raise PipelineError(
                f"Dataset source {source_name} must use "
                f"{expected_slug}/versions/{expected_version}"
            )
        specs.append(
            SourceSpec(
                source_dataset=source_name,
                root=Path(values["root"]),
                layout=values["layout"],
            )
        )
    return tuple(specs)


def _split_config(config: dict) -> SplitConfig:
    values = config["dataset"].get("split", {})
    return SplitConfig(
        seed=int(values.get("seed", 42)),
        train=float(values.get("train", 0.70)),
        val=float(values.get("val", 0.15)),
        test=float(values.get("test", 0.15)),
    )


def run_pipeline(
    *,
    config_path: Path,
    raw_root: Path,
    metadata_root: Path,
    processed_root: Path,
    report_root: Path,
) -> PipelineResult:
    config = _load_config(config_path)
    dataset_config = config["dataset"]
    configured_version = str(dataset_config.get("version", ""))
    if configured_version and processed_root.name != configured_version:
        raise PipelineError(
            f"Processed path version {processed_root.name!r} does not match "
            f"config version {configured_version!r}"
        )
    mapping_path = Path(dataset_config["mapping_path"])
    threshold = int(
        dataset_config.get("duplicates", {}).get("phash_hamming_threshold", 4)
    )
    split_config = _split_config(config)

    raw_manifest = metadata_root / "raw_manifest.csv"
    merge_sources(_source_specs(config), raw_root, raw_manifest)
    cleaning = run_cleaning(
        raw_manifest,
        mapping_path,
        metadata_root,
        phash_threshold=threshold,
    )
    split_manifest = metadata_root / "split_manifest.csv"
    run_split(
        cleaning.scanned_manifest,
        processed_root,
        split_manifest,
        split_config,
    )

    split_records = read_manifest(split_manifest)
    split_bytes = json.dumps(
        {
            "seed": split_config.seed,
            "train": split_config.train,
            "val": split_config.val,
            "test": split_config.test,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    fingerprint = fingerprint_manifest(
        split_records,
        mapping_path.read_bytes(),
        split_bytes,
    )
    validation = validate_processed_dataset(
        processed_root,
        split_manifest,
        phash_threshold=threshold,
    )
    require_valid_processed_dataset(validation)

    class_weights = metadata_root / "class_weights.json"
    write_class_weights(processed_root / "train", class_weights)
    assigned_split = {record.image_id: record.split for record in split_records}
    audited_records = [
        replace(record, split=assigned_split.get(record.image_id, ""))
        for record in read_manifest(cleaning.scanned_manifest)
    ]
    eda_json, eda_text = write_eda_report(
        audited_records,
        fingerprint,
        report_root,
    )
    return PipelineResult(
        processed_root=processed_root,
        split_manifest=split_manifest,
        fingerprint=fingerprint,
        validation=validation,
        class_weights=class_weights,
        eda_json=eda_json,
        eda_text=eda_text,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/preprocessing_config.yaml"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--metadata-root", type=Path, default=Path("data/metadata/v1"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--report-root", type=Path, default=Path("outputs/data/v1"))
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        result = run_pipeline(
            config_path=arguments.config,
            raw_root=arguments.raw_root,
            metadata_root=arguments.metadata_root,
            processed_root=arguments.processed_root,
            report_root=arguments.report_root,
        )
    except PipelineError as error:
        print(f"Dataset pipeline failed: {error}")
        return 2
    print(
        json.dumps(
            {
                "processed_root": str(result.processed_root),
                "split_manifest": str(result.split_manifest),
                "dataset_fingerprint": result.fingerprint,
                "zero_leakage": result.validation.is_valid,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
