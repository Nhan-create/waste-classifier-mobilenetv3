"""Materialize deterministic train/validation/test directories."""

from pathlib import Path

from src.data.manifest import read_manifest
from src.data.split import SplitConfig, assign_splits, materialize_splits


def run_split(
    scanned_manifest_path: Path,
    version_root: Path,
    split_manifest_path: Path,
    config: SplitConfig,
) -> Path:
    """Assign accepted records and materialize one immutable dataset version."""

    records = read_manifest(scanned_manifest_path)
    splits = assign_splits(records, config)
    materialize_splits(splits, version_root, split_manifest_path)
    return split_manifest_path
