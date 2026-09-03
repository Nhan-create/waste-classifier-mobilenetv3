# 10-Class Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, leakage-safe pipeline that combines the two pinned Kaggle datasets into train/validation/test trees containing exactly the ten canonical waste classes.

**Architecture:** `src/data` owns the taxonomy, manifest schema, fingerprint, split contract, and validation rules. Existing scripts under `src/preprocessing` become thin CLI/orchestration layers that ingest, audit, split, and report data while preserving provenance. Training code consumes only a validated versioned dataset and ordered class-weight metadata.

**Tech Stack:** Python 3.10+, Pillow, ImageHash, pandas, PyYAML, NumPy, torchvision, pytest

**Spec:** `docs/superpowers/specs/2026-09-03-mobilenetv3-10-class-design.md`

## Global Constraints

- Dataset sources are `mrgetshjtdone/vn-trash-classification`, version 1, and `sumn2u/garbage-classification-v2`, version 12; both are recorded as MIT-licensed sources.
- Canonical class order is exactly `battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash`.
- Split ratios are train `0.70`, validation `0.15`, test `0.15`, stratified by `source_dataset × unified_label`, with seed `42`.
- A duplicate cluster must never cross a split; pHash Hamming threshold is configurable and defaults to `4`.
- Every split must contain all ten canonical classes or the pipeline exits with a specific error.
- Outputs are written only into a new version directory; existing output is never deleted, merged, or overwritten silently.
- Physical oversampling is disabled; class weights are computed only from the train split and ordered by canonical class names.
- Dataset images, generated manifests, Kaggle credentials, and checkpoints remain untracked; `data/metadata/label_mapping.csv` remains tracked.
- All file writes that define a dataset version are atomic, and all record ordering is deterministic.

---

## File Structure

- `src/data/schema.py`: canonical labels, manifest record, mapping loader, and domain errors.
- `src/data/manifest.py`: deterministic manifest CSV I/O and dataset fingerprinting.
- `src/data/dedup.py`: image audit, SHA-256/pHash calculation, and duplicate clustering.
- `src/data/split.py`: deterministic group-stratified assignment and versioned materialization.
- `src/data/validation.py`: post-split class and leakage verification.
- `src/data/dataset.py`: train-facing class discovery, counts, and ordered class weights.
- `src/preprocessing/*.py`: command-line entry points using the shared contracts above.
- `data/metadata/label_mapping.csv`: the committed source of truth for label mapping.

### Task 1: Canonical taxonomy, mapping, and manifest contract

**Files:**
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/data/schema.py`
- Create: `src/data/manifest.py`
- Create: `data/metadata/label_mapping.csv`
- Modify: `configs/preprocessing_config.yaml`
- Test: `tests/data/test_schema.py`
- Test: `tests/data/test_manifest.py`

**Interfaces:**
- Consumes: no project-local interface.
- Produces: `CLASS_NAMES: tuple[str, ...]`, `ManifestRecord`, `load_label_mapping(path: Path)`, `map_to_unified_label(source_dataset: str, original_label: str, mapping)`, `read_manifest(path: Path)`, `write_manifest(path: Path, records)`, and `fingerprint_manifest(records, mapping_bytes, split_config_bytes)`.

- [ ] **Step 1: Write the failing taxonomy and mapping tests**

```python
from pathlib import Path

import pytest

from src.data.schema import CLASS_NAMES, LabelMappingError, load_label_mapping, map_to_unified_label


def test_canonical_class_order_is_fixed():
    assert CLASS_NAMES == (
        "battery", "biological", "cardboard", "clothes", "glass",
        "metal", "paper", "plastic", "shoes", "trash",
    )


def test_vn_mapping_covers_all_source_labels():
    mapping = load_label_mapping(Path("data/metadata/label_mapping.csv"))
    expected = {
        "Alu": "metal", "Carton": "cardboard", "Foam_box": "plastic",
        "Milk_box": "cardboard", "Other": "trash", "PET": "plastic",
        "Paper": "paper", "Paper_cup": "paper", "Plastic_cup": "plastic",
    }
    assert {label: map_to_unified_label("vn_trash", label, mapping) for label in expected} == expected


def test_unknown_vn_label_is_rejected():
    mapping = load_label_mapping(Path("data/metadata/label_mapping.csv"))
    with pytest.raises(LabelMappingError, match="Mystery"):
        map_to_unified_label("vn_trash", "Mystery", mapping)
```

- [ ] **Step 2: Run the taxonomy test and confirm the missing-module failure**

Run: `pytest tests/data/test_schema.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.data'`.

- [ ] **Step 3: Implement the label contract and committed mapping**

```python
# src/data/schema.py
from dataclasses import dataclass
from pathlib import Path
import csv
import re

CLASS_NAMES = (
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
)


class PipelineError(RuntimeError):
    """A deterministic dataset build cannot continue."""


class LabelMappingError(PipelineError):
    """A source label has no valid canonical mapping."""


def canonicalize_label(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[-\s]+", "_", value.strip().lower()))


def load_label_mapping(path: Path) -> dict[tuple[str, str], str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_dataset", "original_label", "unified_label"}
    if not rows or set(rows[0]) != required:
        raise LabelMappingError(f"Invalid label mapping header in {path}")
    result: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (canonicalize_label(row["source_dataset"]), canonicalize_label(row["original_label"]))
        value = canonicalize_label(row["unified_label"])
        if value not in CLASS_NAMES or key in result:
            raise LabelMappingError(f"Invalid or duplicate mapping row: {row}")
        result[key] = value
    return result


def map_to_unified_label(source_dataset: str, original_label: str,
                         mapping: dict[tuple[str, str], str]) -> str:
    source = canonicalize_label(source_dataset)
    label = canonicalize_label(original_label)
    if source == "garbage_v2" and label in CLASS_NAMES:
        return label
    try:
        return mapping[(source, label)]
    except KeyError as error:
        raise LabelMappingError(f"Unmapped label {original_label!r} for {source_dataset!r}") from error


@dataclass(frozen=True)
class ManifestRecord:
    image_id: str
    source_dataset: str
    original_label: str
    original_split: str
    source_path: str
    raw_path: str
    extension: str
    unified_label: str = ""
    sha256: str = ""
    phash: str = ""
    width: int = 0
    height: int = 0
    mode: str = ""
    status: str = "pending"
    exclusion_reason: str = ""
    cluster_id: str = ""
    duplicate_kind: str = ""
    split: str = ""
```

```csv
source_dataset,original_label,unified_label
vn_trash,Alu,metal
vn_trash,Carton,cardboard
vn_trash,Foam_box,plastic
vn_trash,Milk_box,cardboard
vn_trash,Other,trash
vn_trash,PET,plastic
vn_trash,Paper,paper
vn_trash,Paper_cup,paper
vn_trash,Plastic_cup,plastic
```

Add these exact configuration values under `dataset` in `configs/preprocessing_config.yaml`: `version: v1`, the two slug/version pairs, split ratios `0.70/0.15/0.15`, `seed: 42`, and `phash_hamming_threshold: 4`.

- [ ] **Step 4: Write and run manifest determinism tests**

```python
from dataclasses import replace

from src.data.manifest import fingerprint_manifest
from src.data.schema import ManifestRecord


def test_fingerprint_ignores_record_order_and_tracks_mapping():
    one = ManifestRecord("1", "vn_trash", "PET", "train", "a/PET/1.jpg", "raw/1.jpg", ".jpg")
    two = replace(one, image_id="2", source_path="a/PET/2.jpg", raw_path="raw/2.jpg")
    first = fingerprint_manifest([one, two], b"mapping-a", b'{"seed":42}')
    assert first == fingerprint_manifest([two, one], b"mapping-a", b'{"seed":42}')
    assert first != fingerprint_manifest([one, two], b"mapping-b", b'{"seed":42}')
```

Run: `pytest tests/data/test_schema.py tests/data/test_manifest.py -v`

Expected: FAIL because `src.data.manifest` does not exist, then PASS after implementing CSV I/O with `dataclasses.fields(ManifestRecord)`, sorted `image_id` records, canonical JSON separators, UTF-8, and SHA-256 over length-prefixed manifest/mapping/split-config byte sections.

- [ ] **Step 5: Commit the contract**

```powershell
git add src/__init__.py src/data data/metadata/label_mapping.csv configs/preprocessing_config.yaml tests/data
git commit -m "feat(data): define 10-class manifest and label contract"
```

### Task 2: Deterministic source ingest with provenance

**Files:**
- Modify: `src/preprocessing/merge_datasets.py`
- Test: `tests/preprocessing/test_merge_datasets.py`

**Interfaces:**
- Consumes: `ManifestRecord`, `write_manifest`, and `PipelineError` from Task 1.
- Produces: `SourceSpec`, `stable_image_id(...) -> str`, and `merge_sources(sources, raw_root, manifest_path) -> list[ManifestRecord]`.

- [ ] **Step 1: Write a failing collision/provenance test**

```python
from pathlib import Path

from PIL import Image

from src.preprocessing.merge_datasets import SourceSpec, merge_sources


def test_merge_preserves_source_and_does_not_collide(tmp_path: Path):
    vn = tmp_path / "vn"
    other = tmp_path / "other"
    for root, folder, color in ((vn, "Train/PET", "red"), (other, "plastic", "blue")):
        path = root / folder / "same.jpg"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), color).save(path)
    records = merge_sources(
        [SourceSpec("vn_trash", vn, "vn_trash"), SourceSpec("garbage_v2", other, "garbage_v2")],
        tmp_path / "raw", tmp_path / "raw_manifest.csv",
    )
    assert {record.source_dataset for record in records} == {"vn_trash", "garbage_v2"}
    assert len({record.raw_path for record in records}) == 2
    assert {Path(record.raw_path).read_bytes() for record in records} == {
        (vn / "Train/PET/same.jpg").read_bytes(),
        (other / "plastic/same.jpg").read_bytes(),
    }
```

- [ ] **Step 2: Run the test and verify the interface is missing**

Run: `pytest tests/preprocessing/test_merge_datasets.py::test_merge_preserves_source_and_does_not_collide -v`

Expected: FAIL because `SourceSpec` is not exported.

- [ ] **Step 3: Implement deterministic discovery and copying**

```python
@dataclass(frozen=True)
class SourceSpec:
    source_dataset: str
    root: Path
    layout: Literal["vn_trash", "garbage_v2"]


def stable_image_id(source_dataset: str, original_split: str,
                    relative_path: PurePosixPath) -> str:
    identity = "\0".join((source_dataset, original_split, relative_path.as_posix()))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
```

Implement `merge_sources` by iterating sorted paths with extensions `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp`. VN Trash must read `Train/<label>` and `Test/<label>` and record the lower-case original split; Garbage v2 must accept either a class-root tree or pre-existing train/test/valid roots. Copy to `raw_root/<source_dataset>/<original_label>/<image_id><lowercase_suffix>`, fail if a destination contains different bytes, sort records by `image_id`, and atomically write the raw manifest.

- [ ] **Step 4: Prove reruns are deterministic and missing roots fail clearly**

```python
def test_merge_rerun_has_identical_manifest_bytes(source_fixture, tmp_path):
    manifest = tmp_path / "raw_manifest.csv"
    merge_sources(source_fixture, tmp_path / "raw", manifest)
    first = manifest.read_bytes()
    merge_sources(source_fixture, tmp_path / "raw", manifest)
    assert manifest.read_bytes() == first


def test_missing_source_names_expected_path(tmp_path):
    missing = tmp_path / "vn"
    with pytest.raises(PipelineError, match=re.escape(str(missing))):
        merge_sources([SourceSpec("vn_trash", missing, "vn_trash")], tmp_path / "raw", tmp_path / "m.csv")
```

Run: `pytest tests/preprocessing/test_merge_datasets.py -v`

Expected: PASS.

- [ ] **Step 5: Commit source ingest**

```powershell
git add src/preprocessing/merge_datasets.py tests/preprocessing/test_merge_datasets.py
git commit -m "feat(data): ingest dataset sources with stable provenance"
```

### Task 3: Image audit and duplicate-cluster quarantine

**Files:**
- Create: `src/data/dedup.py`
- Modify: `src/preprocessing/clean_data.py`
- Test: `tests/data/test_dedup.py`
- Test: `tests/preprocessing/test_clean_data.py`

**Interfaces:**
- Consumes: raw `ManifestRecord` rows and `map_to_unified_label` from Task 1.
- Produces: `sha256_file`, `phash_bits`, `hamming_distance`, `audit_raw_manifest(...) -> tuple[list[ManifestRecord], DuplicateReport]`, and three deterministic reports.

- [ ] **Step 1: Write failing exact, near, conflict, and corrupt-image tests**

```python
def test_audit_quarantines_conflicting_duplicate_labels(raw_records, mapping):
    audited, report = audit_raw_manifest(raw_records, mapping, phash_threshold=4)
    assert len(report.conflict_cluster_ids) == 1
    assert {row.status for row in audited} == {"conflict"}
    assert {row.exclusion_reason for row in audited} == {"conflicting_labels"}


def test_hamming_threshold_is_inclusive():
    assert hamming_distance(0b0000, 0b1111) == 4
    assert are_near_duplicates(0b0000, 0b1111, threshold=4)
    assert not are_near_duplicates(0b0000, 0b1111, threshold=3)


def test_corrupt_image_records_specific_reason(corrupt_record, mapping):
    audited, _ = audit_raw_manifest([corrupt_record], mapping, phash_threshold=4)
    assert (audited[0].status, audited[0].exclusion_reason) == ("excluded", "corrupt_image")
```

- [ ] **Step 2: Run the audit tests and verify failure**

Run: `pytest tests/data/test_dedup.py tests/preprocessing/test_clean_data.py -v`

Expected: FAIL because `src.data.dedup` is missing.

- [ ] **Step 3: Implement hashes, union-find clustering, and deterministic selection**

```python
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


def are_near_duplicates(left: int, right: int, threshold: int) -> bool:
    return hamming_distance(left, right) <= threshold
```

Audit each image with `Image.verify()`, reopen it, convert to RGB, and record width, height, mode, extension, SHA-256, and 64-bit pHash. Union identical SHA-256 values first, then union pHashes within the configured Hamming distance using a BK-tree candidate search. Derive `cluster_id` from SHA-256 of sorted member `image_id` values. A cluster with more than one non-empty canonical label marks every row `conflict/conflicting_labels`; otherwise accept the lexicographically first `(source_dataset, original_split, source_path)` row and mark later members `excluded/duplicate_exact` or `excluded/duplicate_near`.

- [ ] **Step 4: Implement and test report files**

`clean_data.py` must atomically write `scanned_manifest.csv`, `duplicate_clusters.csv`, and `label_conflicts.csv` under `data/metadata/<version>/`. `duplicate_clusters.csv` has `cluster_id,image_id,duplicate_kind,status`; `label_conflicts.csv` has `cluster_id,image_id,source_dataset,original_label,unified_label`.

```python
def test_clean_command_writes_separate_reports(clean_fixture, tmp_path):
    output = tmp_path / "metadata"
    result = run_cleaning(clean_fixture.manifest, clean_fixture.mapping, output, phash_threshold=4)
    assert result.scanned_manifest == output / "scanned_manifest.csv"
    assert (output / "duplicate_clusters.csv").is_file()
    assert (output / "label_conflicts.csv").is_file()
```

Run: `pytest tests/data/test_dedup.py tests/preprocessing/test_clean_data.py -v`

Expected: PASS and no test equates pHash equality with all near duplicates.

- [ ] **Step 5: Commit image auditing**

```powershell
git add src/data/dedup.py src/preprocessing/clean_data.py tests/data/test_dedup.py tests/preprocessing/test_clean_data.py
git commit -m "feat(data): audit images and quarantine duplicate clusters"
```

### Task 4: Group-stratified split and immutable materialization

**Files:**
- Create: `src/data/split.py`
- Modify: `src/preprocessing/split_dataset.py`
- Test: `tests/data/test_split.py`
- Test: `tests/preprocessing/test_split_dataset.py`

**Interfaces:**
- Consumes: accepted audited records from Task 3 and `CLASS_NAMES` from Task 1.
- Produces: `SplitConfig`, `assign_splits(records, config) -> dict[str, list[ManifestRecord]]`, and `materialize_splits(splits, version_root, manifest_path)`.

- [ ] **Step 1: Write failing grouped/deterministic split tests**

```python
def test_group_split_is_deterministic_and_keeps_clusters_together(records):
    config = SplitConfig(seed=42, train=0.70, val=0.15, test=0.15)
    first = assign_splits(records, config)
    second = assign_splits(list(reversed(records)), config)
    first_index = {row.image_id: split for split, rows in first.items() for row in rows}
    second_index = {row.image_id: split for split, rows in second.items() for row in rows}
    assert first_index == second_index
    by_cluster = defaultdict(set)
    for row_id, split in first_index.items():
        by_cluster[next(row.cluster_id for row in records if row.image_id == row_id)].add(split)
    assert all(len(splits) == 1 for splits in by_cluster.values())


def test_each_split_contains_exactly_all_classes(records):
    result = assign_splits(records, SplitConfig())
    assert {name: {row.unified_label for row in rows} for name, rows in result.items()} == {
        "train": set(CLASS_NAMES), "val": set(CLASS_NAMES), "test": set(CLASS_NAMES),
    }
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest tests/data/test_split.py -v`

Expected: FAIL because `src.data.split` is missing.

- [ ] **Step 3: Implement deterministic group allocation**

```python
@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def ratios(self) -> dict[str, float]:
        values = {"train": self.train, "val": self.val, "test": self.test}
        if not math.isclose(sum(values.values()), 1.0):
            raise PipelineError(f"Split ratios must sum to 1.0: {values}")
        return values
```

For each `(source_dataset, unified_label)` bucket, group rows by `cluster_id`, sort by a stable `sha256(f"{seed}:{cluster_id}")` key, seed each of train/val/test with one group, then place each remaining group into the split that minimizes absolute deviation from the target image counts. Reject any bucket with fewer than three eligible groups and name the source and class in the exception. After all buckets, verify each split's class set exactly equals `CLASS_NAMES`.

- [ ] **Step 4: Implement immutable copy and prove existing outputs survive**

```python
def test_materialize_refuses_existing_version_without_deleting_it(split_fixture, tmp_path):
    version_root = tmp_path / "processed" / "v1"
    version_root.mkdir(parents=True)
    sentinel = version_root / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(PipelineError, match="already exists"):
        materialize_splits(split_fixture, version_root, tmp_path / "split_manifest.csv")
    assert sentinel.read_text(encoding="utf-8") == "keep"
```

Create `<version_root>/<split>/<class_id>/<image_id><extension>` with `copy2`; never remove or reuse a version directory. Write `split_manifest.csv` atomically only after all copies succeed, including the assigned `split` field.

Run: `pytest tests/data/test_split.py tests/preprocessing/test_split_dataset.py -v`

Expected: PASS.

- [ ] **Step 5: Commit split behavior**

```powershell
git add src/data/split.py src/preprocessing/split_dataset.py tests/data/test_split.py tests/preprocessing/test_split_dataset.py
git commit -m "feat(data): create deterministic group-safe dataset splits"
```

### Task 5: Post-split leakage gate and ordered class weights

**Files:**
- Create: `src/data/validation.py`
- Create: `src/data/dataset.py`
- Modify: `src/preprocessing/balance_classes.py`
- Modify: `src/preprocessing/dataloader.py`
- Modify: `src/preprocessing/eda.py`
- Test: `tests/data/test_validation.py`
- Test: `tests/data/test_dataset.py`

**Interfaces:**
- Consumes: versioned split tree and `split_manifest.csv` from Task 4.
- Produces: `ValidationReport`, `validate_processed_dataset(...)`, `discover_class_names(...)`, `compute_class_counts(...)`, `compute_class_weights(...)`, and `make_imagefolder_dataloader(...)`.

- [ ] **Step 1: Write failing cross-split leakage and class-set tests**

```python
def test_validator_reports_exact_and_near_leakage_separately(leaky_dataset):
    report = validate_processed_dataset(
        leaky_dataset.root, leaky_dataset.manifest, CLASS_NAMES, phash_threshold=4,
    )
    assert report.exact_cross_split_clusters == (leaky_dataset.exact_cluster,)
    assert report.near_cross_split_clusters == (leaky_dataset.near_cluster,)
    assert report.is_valid is False


def test_discovery_rejects_missing_class(tmp_path):
    for class_name in CLASS_NAMES[:-1]:
        (tmp_path / class_name).mkdir()
    with pytest.raises(PipelineError, match="trash"):
        discover_class_names(tmp_path)
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest tests/data/test_validation.py tests/data/test_dataset.py -v`

Expected: FAIL because the validation and dataset modules are missing.

- [ ] **Step 3: Implement the zero-leakage validation gate**

```python
@dataclass(frozen=True)
class ValidationReport:
    class_counts: dict[str, dict[str, int]]
    exact_cross_split_clusters: tuple[str, ...]
    near_cross_split_clusters: tuple[str, ...]
    missing_classes: dict[str, tuple[str, ...]]

    @property
    def is_valid(self) -> bool:
        return not (
            self.exact_cross_split_clusters
            or self.near_cross_split_clusters
            or any(self.missing_classes.values())
        )
```

Recompute SHA-256 and pHash for every materialized file. Build exact and near duplicate clusters across all splits, report cluster IDs that touch more than one split, and compare each split directory set with `CLASS_NAMES`. The CLI exits `0` only when `report.is_valid`; otherwise it prints sorted defects and exits `2`. Training must call this gate before creating the model.

- [ ] **Step 4: Implement ordered counts/weights and remove physical oversampling**

```python
def compute_class_weights(train_dir: Path,
                          class_names: Sequence[str] = CLASS_NAMES) -> dict[str, float]:
    counts = compute_class_counts(train_dir, class_names)
    total = sum(counts.values())
    if any(count == 0 for count in counts.values()):
        missing = [name for name, count in counts.items() if count == 0]
        raise PipelineError(f"Cannot compute class weights; empty classes: {missing}")
    return {name: total / (len(class_names) * counts[name]) for name in class_names}


def ordered_weight_tensor(weights: Mapping[str, float], class_names: Sequence[str]) -> torch.Tensor:
    return torch.tensor([weights[name] for name in class_names], dtype=torch.float32)
```

Delete the command path that creates `train_oversampled`; `balance_classes.py` writes only ordered `class_weights.json`. `dataloader.py` must use `ImageFolder`, verify `dataset.classes == list(CLASS_NAMES)`, use augmentation group C only for train, and use only resize, tensor conversion, and ImageNet normalization for validation/test. `eda.py` writes `outputs/data/<version>/eda.json` with fingerprint, raw/accepted/excluded counts, exclusion reasons, per-source/class counts, split counts, and exact/near/conflict totals.

- [ ] **Step 5: Run all data-facing tests**

Run: `pytest tests/data tests/preprocessing -v`

Expected: PASS; class-weight tensors stay aligned even if JSON keys are shuffled.

- [ ] **Step 6: Commit the training-facing data gate**

```powershell
git add src/data/validation.py src/data/dataset.py src/preprocessing/balance_classes.py src/preprocessing/dataloader.py src/preprocessing/eda.py tests/data
git commit -m "feat(data): validate leakage and expose ordered class weights"
```

### Task 6: Pipeline CLI, documentation, and data-only acceptance test

**Files:**
- Create: `src/preprocessing/run_pipeline.py`
- Create: `tests/integration/test_data_pipeline.py`
- Modify: `README.md`
- Modify: `Tienxulidulieu.md`
- Modify: `.gitignore`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: all Task 1–5 APIs.
- Produces: one `python -m src.preprocessing.run_pipeline` command and a validated `data/processed/<version>` result with metadata fingerprint.

- [ ] **Step 1: Write a failing synthetic end-to-end acceptance test**

```python
def test_pipeline_builds_valid_ten_class_dataset(synthetic_sources, tmp_path):
    result = run_pipeline(
        config_path=synthetic_sources.config,
        raw_root=tmp_path / "raw",
        metadata_root=tmp_path / "metadata" / "v1",
        processed_root=tmp_path / "processed" / "v1",
    )
    assert result.validation.is_valid
    assert result.fingerprint
    for split in ("train", "val", "test"):
        assert {path.name for path in (result.processed_root / split).iterdir()} == set(CLASS_NAMES)
```

- [ ] **Step 2: Run the acceptance test and verify the missing orchestrator**

Run: `pytest tests/integration/test_data_pipeline.py -v`

Expected: FAIL because `run_pipeline` is not defined.

- [ ] **Step 3: Implement orchestration with explicit stage boundaries**

```python
@dataclass(frozen=True)
class PipelineResult:
    processed_root: Path
    split_manifest: Path
    fingerprint: str
    validation: ValidationReport


def run_pipeline(config_path: Path, raw_root: Path, metadata_root: Path,
                 processed_root: Path) -> PipelineResult:
    config = load_config(config_path)
    raw = merge_sources(source_specs(config), raw_root, metadata_root / "raw_manifest.csv")
    audited, duplicate_report = audit_raw_manifest(raw, load_label_mapping(config.mapping_path),
                                                   phash_threshold=config.phash_threshold)
    write_audit_outputs(audited, duplicate_report, metadata_root)
    splits = assign_splits(audited, config.split)
    split_manifest = metadata_root / "split_manifest.csv"
    materialize_splits(splits, processed_root, split_manifest)
    fingerprint = fingerprint_from_files(split_manifest, config.mapping_path, config.split)
    validation = validate_processed_dataset(processed_root, split_manifest, CLASS_NAMES,
                                            config.phash_threshold)
    if not validation.is_valid:
        raise PipelineError(f"Post-split validation failed: {validation}")
    return PipelineResult(processed_root, split_manifest, fingerprint, validation)
```

- [ ] **Step 4: Document exact local commands and ignore generated material**

README and `Tienxulidulieu.md` must include this command sequence:

```powershell
python -m src.preprocessing.run_pipeline --config configs/preprocessing_config.yaml
python -m src.data.validation --dataset-root data/processed/v1 --manifest data/metadata/v1/split_manifest.csv
```

Add `kagglehub` and test/runtime dependencies to `requirements.txt`. Ignore `data/raw/`, `data/interim/`, `data/processed/`, `data/metadata/*/`, `artifacts/`, `.kaggle/`, `kaggle.json`, and runtime SQLite files while explicitly unignoring `data/metadata/label_mapping.csv`.

- [ ] **Step 5: Verify the complete data plan**

Run: `python -m compileall src`

Expected: exit code `0`.

Run: `pytest tests/data tests/preprocessing tests/integration/test_data_pipeline.py -q`

Expected: all tests PASS using generated temporary images and no Kaggle download.

- [ ] **Step 6: Commit the data pipeline**

```powershell
git add src/preprocessing/run_pipeline.py tests/integration/test_data_pipeline.py README.md Tienxulidulieu.md .gitignore requirements.txt
git commit -m "docs(data): document reproducible 10-class dataset build"
```

## Plan Acceptance

- Tasks 1–3 cover pinned-source provenance, the nine-row VN mapping, Pillow validation, SHA-256, configurable pHash clustering, deterministic representatives, and conflict quarantine.
- Tasks 4–5 cover source/class stratification, seed 42, 70/15/15 allocation, duplicate-group integrity, immutable output versions, all-ten-class enforcement, zero-leakage validation, and ordered train-only class weights.
- Task 6 provides a single reproducible command, documentation, ignore rules, and a synthetic acceptance test that requires no real dataset.
- The next plan consumes only `data/processed/<version>`, `split_manifest.csv`, `class_weights.json`, `CLASS_NAMES`, and the dataset fingerprint.
