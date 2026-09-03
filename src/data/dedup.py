"""Image validation and exact/near duplicate clustering."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

from .schema import LabelMappingError, ManifestRecord, map_to_unified_label


@dataclass(frozen=True)
class ImageSignature:
    index: int
    sha256: str
    phash: int


@dataclass(frozen=True)
class DuplicateReport:
    exact_cluster_count: int
    near_cluster_count: int
    conflict_cluster_ids: tuple[str, ...]


class _UnionFind:
    def __init__(self, keys: Iterable[int]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: int) -> int:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            parent = self.parent[key]
            self.parent[key] = root
            key = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


@dataclass
class _BKNode:
    value: int
    indexes: list[int]
    children: dict[int, _BKNode]


class _BKTree:
    def __init__(self) -> None:
        self.root: _BKNode | None = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = _BKNode(value=value, indexes=[index], children={})
            return
        node = self.root
        while True:
            distance = hamming_distance(value, node.value)
            if distance == 0:
                node.indexes.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(
                    value=value,
                    indexes=[index],
                    children={},
                )
                return
            node = child

    def query(self, value: int, maximum_distance: int) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = hamming_distance(value, node.value)
            if distance <= maximum_distance:
                matches.extend(node.indexes)
            minimum = distance - maximum_distance
            maximum = distance + maximum_distance
            pending.extend(
                child
                for edge, child in node.children.items()
                if minimum <= edge <= maximum
            )
        return matches


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash_bits(image: Image.Image) -> int:
    return int(str(imagehash.phash(image.convert("RGB"))), 16)


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def find_duplicate_clusters(
    signatures: Sequence[ImageSignature],
    phash_threshold: int,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic transitive clusters for exact SHA and near pHash."""

    if not 0 <= phash_threshold <= 64:
        raise ValueError(
            f"phash_threshold must be between 0 and 64, received {phash_threshold}"
        )
    indexes = [signature.index for signature in signatures]
    if len(indexes) != len(set(indexes)):
        raise ValueError("Image signature indexes must be unique")
    union_find = _UnionFind(indexes)
    first_by_sha: dict[str, int] = {}
    tree = _BKTree()
    for signature in sorted(signatures, key=lambda item: item.index):
        previous = first_by_sha.setdefault(signature.sha256, signature.index)
        union_find.union(previous, signature.index)
        for candidate in tree.query(signature.phash, phash_threshold):
            union_find.union(candidate, signature.index)
        tree.add(signature.phash, signature.index)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in indexes:
        grouped[union_find.find(index)].append(index)
    clusters = [tuple(sorted(group)) for group in grouped.values()]
    return tuple(sorted(clusters, key=lambda group: group[0]))


def _cluster_id(records: Sequence[ManifestRecord]) -> str:
    joined = "\0".join(sorted(record.image_id for record in records))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def _inspect_record(
    record: ManifestRecord,
    mapping: Mapping[tuple[str, str], str],
) -> tuple[ManifestRecord, ImageSignature | None]:
    path = Path(record.raw_path)
    if not path.is_file():
        return replace(
            record,
            status="excluded",
            exclusion_reason="missing_file",
        ), None

    file_hash = sha256_file(path)
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            perceptual_hash = phash_bits(image)
    except (UnidentifiedImageError, OSError, ValueError):
        return replace(
            record,
            sha256=file_hash,
            status="excluded",
            exclusion_reason="corrupt_image",
        ), None

    try:
        unified_label = map_to_unified_label(
            record.source_dataset,
            record.original_label,
            mapping,
        )
        status = "candidate"
        exclusion_reason = ""
    except LabelMappingError:
        unified_label = ""
        status = "excluded"
        exclusion_reason = "unmapped_label"

    inspected = replace(
        record,
        unified_label=unified_label,
        sha256=file_hash,
        phash=f"{perceptual_hash:016x}",
        width=width,
        height=height,
        mode=mode,
        status=status,
        exclusion_reason=exclusion_reason,
    )
    return inspected, ImageSignature(-1, file_hash, perceptual_hash)


def audit_raw_manifest(
    raw_records: Sequence[ManifestRecord],
    mapping: Mapping[tuple[str, str], str],
    *,
    phash_threshold: int,
) -> tuple[list[ManifestRecord], DuplicateReport]:
    """Validate images, assign labels, and quarantine duplicate conflicts."""

    audited: list[ManifestRecord] = []
    signatures: list[ImageSignature] = []
    for record in sorted(raw_records, key=lambda item: item.image_id):
        inspected, signature = _inspect_record(record, mapping)
        index = len(audited)
        audited.append(inspected)
        if signature is not None:
            signatures.append(replace(signature, index=index))

    exact_clusters = 0
    near_clusters = 0
    conflicts: list[str] = []
    for indexes in find_duplicate_clusters(signatures, phash_threshold):
        members = [audited[index] for index in indexes]
        cluster_id = _cluster_id(members)
        sha_counts = Counter(member.sha256 for member in members)
        has_exact = any(count > 1 for count in sha_counts.values())
        has_near = len({member.sha256 for member in members}) > 1
        if len(members) > 1:
            exact_clusters += int(has_exact)
            near_clusters += int(has_near)

        labels = {member.unified_label for member in members if member.unified_label}
        if len(labels) > 1:
            conflicts.append(cluster_id)
            for index in indexes:
                audited[index] = replace(
                    audited[index],
                    cluster_id=cluster_id,
                    status="conflict",
                    exclusion_reason="conflicting_labels",
                )
            continue

        candidates = [
            (index, audited[index])
            for index in indexes
            if audited[index].status == "candidate"
        ]
        representative_index = None
        representative = None
        if candidates:
            representative_index, representative = min(
                candidates,
                key=lambda item: (
                    item[1].source_dataset,
                    item[1].original_split,
                    item[1].source_path,
                    item[1].image_id,
                ),
            )

        for index in indexes:
            member = audited[index]
            if index == representative_index:
                audited[index] = replace(
                    member,
                    cluster_id=cluster_id,
                    status="accepted",
                    exclusion_reason="",
                    duplicate_kind="",
                )
            elif member.status == "candidate" and representative is not None:
                duplicate_kind = (
                    "exact" if member.sha256 == representative.sha256 else "near"
                )
                audited[index] = replace(
                    member,
                    cluster_id=cluster_id,
                    status="excluded",
                    exclusion_reason=f"duplicate_{duplicate_kind}",
                    duplicate_kind=duplicate_kind,
                )
            else:
                audited[index] = replace(member, cluster_id=cluster_id)

    return audited, DuplicateReport(
        exact_cluster_count=exact_clusters,
        near_cluster_count=near_clusters,
        conflict_cluster_ids=tuple(sorted(conflicts)),
    )
