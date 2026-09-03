"""Deterministic manifest persistence and fingerprinting."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, fields
from pathlib import Path

from .schema import ManifestRecord, PipelineError

_FIELD_NAMES = tuple(field.name for field in fields(ManifestRecord))
_INTEGER_FIELDS = {"width", "height"}


def _atomic_replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    temporary.replace(path)


def write_manifest(path: Path, records: Iterable[ManifestRecord]) -> None:
    """Write records sorted by stable image ID using an atomic replace."""

    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_FIELD_NAMES, lineterminator="\n")
    writer.writeheader()
    for record in sorted(records, key=lambda item: item.image_id):
        writer.writerow(asdict(record))
    _atomic_replace_text(path, buffer.getvalue())


def read_manifest(path: Path) -> list[ManifestRecord]:
    """Load a manifest and reject incompatible columns."""

    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != _FIELD_NAMES:
                raise PipelineError(
                    f"Invalid manifest header in {path}; expected {list(_FIELD_NAMES)}"
                )
            records: list[ManifestRecord] = []
            for row in reader:
                values: dict[str, object] = dict(row)
                for name in _INTEGER_FIELDS:
                    values[name] = int(row[name] or 0)
                records.append(ManifestRecord(**values))
    except FileNotFoundError as error:
        raise PipelineError(f"Manifest not found: {path}") from error
    return sorted(records, key=lambda item: item.image_id)


def _feed_section(digest: hashlib._Hash, data: bytes) -> None:
    digest.update(len(data).to_bytes(8, byteorder="big", signed=False))
    digest.update(data)


def fingerprint_manifest(
    records: Iterable[ManifestRecord],
    mapping_bytes: bytes,
    split_config_bytes: bytes,
) -> str:
    """Hash canonical manifest content plus mapping and split configuration."""

    canonical_rows = [
        asdict(record) for record in sorted(records, key=lambda item: item.image_id)
    ]
    manifest_bytes = json.dumps(
        canonical_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    for section in (manifest_bytes, mapping_bytes, split_config_bytes):
        _feed_section(digest, section)
    return digest.hexdigest()
