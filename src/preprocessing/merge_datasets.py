"""Ingest source datasets into a collision-safe raw tree."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from src.data.manifest import write_manifest
from src.data.schema import VALID_IMAGE_EXTENSIONS, ManifestRecord, PipelineError

DatasetLayout = Literal["vn_trash", "garbage_v2"]


@dataclass(frozen=True)
class SourceSpec:
    """A configured source dataset and its directory layout."""

    source_dataset: str
    root: Path
    layout: DatasetLayout


def stable_image_id(
    source_dataset: str,
    original_split: str,
    relative_path: PurePosixPath,
) -> str:
    """Build a stable ID from provenance rather than mutable file contents."""

    identity = f"{source_dataset}\0{original_split}\0{relative_path.as_posix()}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_files(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix().lower()):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            yield path


def _vn_entries(spec: SourceSpec) -> Iterator[tuple[Path, str, str]]:
    for split_name in ("Train", "Test"):
        split_root = spec.root / split_name
        if not split_root.is_dir():
            raise PipelineError(
                f"VN Trash layout requires directory: {split_root}"
            )
        for label_dir in sorted(split_root.iterdir(), key=lambda item: item.name.lower()):
            if not label_dir.is_dir():
                continue
            for image_path in _image_files(label_dir):
                yield image_path, label_dir.name, split_name.lower()


def _garbage_entries(spec: SourceSpec) -> Iterator[tuple[Path, str, str]]:
    original_root = spec.root / "original"
    if original_root.is_dir():
        class_roots = [(original_root, "")]
    else:
        split_aliases = {
            "train": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "test": "test",
        }
        split_roots = [
            (path, split_aliases[path.name.lower()])
            for path in spec.root.iterdir()
            if path.is_dir() and path.name.lower() in split_aliases
        ]
        class_roots = split_roots or [(spec.root, "")]

    for class_root, original_split in sorted(
        class_roots, key=lambda item: item[0].as_posix().lower()
    ):
        for label_dir in sorted(class_root.iterdir(), key=lambda item: item.name.lower()):
            if not label_dir.is_dir():
                continue
            for image_path in _image_files(label_dir):
                yield image_path, label_dir.name, original_split


def _source_entries(spec: SourceSpec) -> Iterator[tuple[Path, str, str]]:
    if not spec.root.is_dir():
        raise PipelineError(f"Dataset source directory not found: {spec.root}")
    if spec.layout == "vn_trash":
        yield from _vn_entries(spec)
        return
    if spec.layout == "garbage_v2":
        yield from _garbage_entries(spec)
        return
    raise PipelineError(f"Unsupported dataset layout: {spec.layout}")


def _copy_without_collision(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _file_sha256(source) != _file_sha256(destination):
            raise PipelineError(
                f"Refusing to overwrite different image at {destination}"
            )
        return
    shutil.copy2(source, destination)


def merge_sources(
    sources: Sequence[SourceSpec] | Iterable[SourceSpec],
    raw_root: Path,
    manifest_path: Path,
) -> list[ManifestRecord]:
    """Copy all supported images and persist deterministic provenance."""

    records: list[ManifestRecord] = []
    for spec in sources:
        for image_path, original_label, original_split in _source_entries(spec):
            relative_path = PurePosixPath(image_path.relative_to(spec.root).as_posix())
            image_id = stable_image_id(
                spec.source_dataset,
                original_split,
                relative_path,
            )
            extension = image_path.suffix.lower()
            destination = (
                raw_root
                / spec.source_dataset
                / original_label
                / f"{image_id}{extension}"
            )
            _copy_without_collision(image_path, destination)
            records.append(
                ManifestRecord(
                    image_id=image_id,
                    source_dataset=spec.source_dataset,
                    original_label=original_label,
                    original_split=original_split,
                    source_path=relative_path.as_posix(),
                    raw_path=str(destination),
                    extension=extension,
                )
            )

    records.sort(key=lambda item: item.image_id)
    write_manifest(manifest_path, records)
    return records
