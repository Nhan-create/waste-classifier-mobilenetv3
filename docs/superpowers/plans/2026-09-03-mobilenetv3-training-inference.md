# MobileNetV3 Training, Evaluation, and Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested MobileNetV3-Large training stack with resumable self-describing checkpoints, ten-class evaluation artifacts, and metadata-safe top-k inference.

**Architecture:** A single model factory is shared by training, evaluation, and prediction. A versioned checkpoint envelope carries the canonical class order, preprocessing contract, dataset fingerprint, and optional resume state; no consumer recreates class indices independently. Training has explicit frozen-head and full-fine-tuning phases, while evaluation and inference are separate read-only consumers.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, scikit-learn, pandas, Pillow, NumPy, matplotlib, seaborn, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-09-03-mobilenetv3-10-class-design.md`

## Global Constraints

- Implement only stable model ID `mobilenet_v3_large`; MobileNetV3-Small and legacy binary-model checkpoint migration are out of scope.
- Model input is RGB `224×224` with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- Model output is ten logits; softmax is applied only during evaluation and inference.
- Canonical class order is exactly `battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash` and is stored in every checkpoint.
- Default batch size is `32`; seed is `42`; deterministic execution is configurable.
- Loss is class-weighted cross-entropy with label smoothing `0.1`.
- Optimizer is AdamW with weight decay `1e-4`; phase 1 trains the head for `5` epochs at `1e-3`; phase 2 trains the backbone at `1e-4` and head at `3e-4` for at most `25` epochs.
- Use cosine learning-rate scheduling, CUDA AMP only when CUDA is active, gradient clipping `1.0`, and early stopping patience `7` on validation macro-F1.
- Select `best.pt` using validation macro-F1 only; evaluate the test split only after training selection is complete.
- A checkpoint class-set/order mismatch, head mismatch, or unknown format version fails before `load_state_dict` can return a mislabeled model.
- Full dataset training is not required on this machine; CPU unit and smoke tests use synthetic data.

---

## File Structure

- `configs/model_config.yaml`: model, transforms, optimization, and inference defaults.
- `src/models/mobilenetv3.py`: the only MobileNetV3 construction point.
- `src/training/checkpoint.py`: checkpoint schema, compatibility checks, atomic persistence, and fingerprint metadata.
- `src/training/data.py`: validated ImageFolder datasets, transforms, and ordered class weights.
- `src/training/engine.py`: phase configuration, epoch execution, macro-F1 selection, and early stopping.
- `src/training/train.py`: CLI orchestration, resume, and artifact layout.
- `src/evaluation/metrics.py`: deterministic multiclass metric calculations.
- `src/evaluation/evaluate.py`: checkpoint/test loading and report generation.
- `src/inference/predict.py`: lazy predictor and JSON CLI.

### Task 1: Model configuration and MobileNetV3-Large factory

**Files:**
- Create: `configs/model_config.yaml`
- Create: `src/models/__init__.py`
- Create: `src/models/mobilenetv3.py`
- Test: `tests/models/test_mobilenetv3.py`
- Test: `tests/models/test_model_config.py`

**Interfaces:**
- Consumes: `CLASS_NAMES` from `src.data.schema` created by the data plan.
- Produces: `MODEL_ID`, `build_model(model_name: str, num_classes: int, pretrained: bool) -> torch.nn.Module`, and the complete YAML defaults used by all later tasks.

- [ ] **Step 1: Write failing factory and configuration tests**

```python
import torch
import yaml

from src.models.mobilenetv3 import MODEL_ID, build_model


def test_factory_returns_ten_logits_without_softmax():
    model = build_model(MODEL_ID, num_classes=10, pretrained=False).eval()
    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 10)
    assert model.classifier[3].out_features == 10
    assert not torch.allclose(logits.sum(dim=1), torch.ones(2), atol=1e-4)


def test_model_config_contains_approved_defaults():
    config = yaml.safe_load(open("configs/model_config.yaml", encoding="utf-8"))
    assert config["model"] == {"name": "mobilenet_v3_large", "pretrained": True, "num_classes": 10}
    assert config["training"]["phase1"] == {"epochs": 5, "head_lr": 0.001}
    assert config["training"]["phase2"] == {"epochs": 25, "backbone_lr": 0.0001, "head_lr": 0.0003}
```

- [ ] **Step 2: Run the tests and verify the missing model module**

Run: `pytest tests/models/test_mobilenetv3.py tests/models/test_model_config.py -v`

Expected: FAIL because `src.models.mobilenetv3` and `configs/model_config.yaml` do not exist.

- [ ] **Step 3: Implement the single model factory**

```python
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

MODEL_ID = "mobilenet_v3_large"


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name != MODEL_ID:
        raise ValueError(f"Unsupported model {model_name!r}; expected {MODEL_ID!r}")
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, received {num_classes}")
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    final = model.classifier[3]
    model.classifier[3] = nn.Linear(final.in_features, num_classes)
    return model
```

Create `configs/model_config.yaml` with the exact global constraint values plus `input_size: 224`, ImageNet mean/std, augmentation group `C`, confidence threshold `0.55`, and checkpoint format version `1`.

- [ ] **Step 4: Run the focused tests**

Run: `pytest tests/models/test_mobilenetv3.py tests/models/test_model_config.py -v`

Expected: PASS without downloading pretrained weights because the test passes `pretrained=False`.

- [ ] **Step 5: Commit the model factory**

```powershell
git add configs/model_config.yaml src/models tests/models
git commit -m "feat(model): add MobileNetV3-Large factory and config"
```

### Task 2: Versioned self-describing checkpoint envelope

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/checkpoint.py`
- Test: `tests/training/test_checkpoint.py`

**Interfaces:**
- Consumes: `build_model`, `MODEL_ID`, and `CLASS_NAMES`.
- Produces: `CheckpointMetadata`, `LoadedCheckpoint`, `CheckpointCompatibilityError`, `save_checkpoint(...)`, `load_checkpoint(...)`, and `build_model_from_checkpoint(...)`.

- [ ] **Step 1: Write failing deploy/resume round-trip tests**

```python
import torch

from src.models.mobilenetv3 import MODEL_ID, build_model
from src.training.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint


def metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        format_version=1,
        model_name=MODEL_ID,
        num_classes=10,
        class_names=("battery", "biological", "cardboard", "clothes", "glass",
                     "metal", "paper", "plastic", "shoes", "trash"),
        input_size=224,
        normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        epoch=4,
        metrics={"val_macro_f1": 0.63},
        dataset_fingerprint="a" * 64,
        training_config={"seed": 42},
    )


def test_best_and_last_use_one_envelope(tmp_path):
    model = build_model(MODEL_ID, 10, pretrained=False)
    best = tmp_path / "best.pt"
    last = tmp_path / "last.pt"
    save_checkpoint(best, metadata(), model.state_dict())
    save_checkpoint(last, metadata(), model.state_dict(), resume_state={"optimizer": {"state": {}}})
    assert load_checkpoint(best, torch.device("cpu")).resume_state is None
    assert load_checkpoint(last, torch.device("cpu")).resume_state == {"optimizer": {"state": {}}}
```

- [ ] **Step 2: Write mismatch rejection tests and run them**

```python
def test_expected_class_order_mismatch_is_rejected(tmp_path):
    model = build_model(MODEL_ID, 10, pretrained=False)
    path = tmp_path / "bad.pt"
    save_checkpoint(path, metadata(), model.state_dict())
    with pytest.raises(CheckpointCompatibilityError, match="class order"):
        load_checkpoint(
            path,
            torch.device("cpu"),
            expected_classes=tuple(reversed(metadata().class_names)),
        )


def test_unknown_format_and_wrong_head_are_rejected(tmp_path):
    model = build_model(MODEL_ID, 10, pretrained=False)
    path = tmp_path / "bad-format.pt"
    envelope = {
        "metadata": dataclasses.asdict(dataclasses.replace(metadata(), format_version=99)),
        "model_state": model.state_dict(),
        "resume_state": None,
    }
    torch.save(envelope, path)
    with pytest.raises(CheckpointCompatibilityError, match="format_version"):
        load_checkpoint(path, torch.device("cpu"))

    wrong_head = tmp_path / "bad-head.pt"
    envelope["metadata"]["format_version"] = 1
    envelope["model_state"]["classifier.3.weight"] = torch.zeros(9, model.classifier[3].in_features)
    torch.save(envelope, wrong_head)
    with pytest.raises(CheckpointCompatibilityError, match="classifier"):
        load_checkpoint(wrong_head, torch.device("cpu"))
```

Run: `pytest tests/training/test_checkpoint.py -v`

Expected: FAIL because the checkpoint module is missing.

- [ ] **Step 3: Implement typed metadata and strict validation**

```python
@dataclass(frozen=True)
class CheckpointMetadata:
    format_version: int
    model_name: str
    num_classes: int
    class_names: tuple[str, ...]
    input_size: int
    normalization: dict[str, list[float]]
    epoch: int
    metrics: dict[str, float | None]
    dataset_fingerprint: str
    training_config: dict[str, object]


@dataclass(frozen=True)
class LoadedCheckpoint:
    metadata: CheckpointMetadata
    model_state: dict[str, torch.Tensor]
    resume_state: dict[str, object] | None


class CheckpointCompatibilityError(RuntimeError):
    """Checkpoint metadata and tensor shapes do not describe one valid model."""
```

Validation must require exactly format version `1`, model ID `mobilenet_v3_large`, `num_classes == len(class_names)`, unique class IDs, valid mean/std of length three, a 64-character hexadecimal fingerprint, and classifier weight/bias first dimension equal to `num_classes`. If `expected_classes` is supplied, require tuple equality and report the first differing index.

- [ ] **Step 4: Implement atomic save/load and model reconstruction**

```python
def save_checkpoint(path: Path, metadata: CheckpointMetadata,
                    model_state: Mapping[str, torch.Tensor],
                    resume_state: Mapping[str, object] | None = None) -> None:
    envelope = {
        "metadata": asdict(metadata),
        "model_state": dict(model_state),
        "resume_state": None if resume_state is None else dict(resume_state),
    }
    validate_envelope(envelope)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(envelope, temporary)
    temporary.replace(path)


def build_model_from_checkpoint(loaded: LoadedCheckpoint, device: torch.device) -> nn.Module:
    model = build_model(loaded.metadata.model_name, loaded.metadata.num_classes, pretrained=False)
    model.load_state_dict(loaded.model_state, strict=True)
    return model.to(device).eval()
```

`load_checkpoint` calls `torch.load(path, map_location=device, weights_only=False)`, constructs the dataclass, validates the envelope before model reconstruction, and returns `LoadedCheckpoint`.

- [ ] **Step 5: Verify exact prediction equivalence after save/load**

Run: `pytest tests/training/test_checkpoint.py -v`

Expected: PASS, including a test comparing source and reloaded eval logits with `torch.testing.assert_close`.

- [ ] **Step 6: Commit checkpoint support**

```powershell
git add src/training tests/training/test_checkpoint.py
git commit -m "feat(training): add self-describing checkpoint contract"
```

### Task 3: Validated loaders, transforms, and class weighting

**Files:**
- Create: `src/training/data.py`
- Modify: `src/preprocessing/dataloader.py`
- Test: `tests/training/test_data.py`

**Interfaces:**
- Consumes: `CLASS_NAMES`, `discover_class_names`, the zero-leakage gate, and `configs/model_config.yaml`.
- Produces: `build_transforms(config, train)`, `create_datasets(data_root, config)`, `create_dataloaders(...)`, and `ordered_class_weight_tensor(train_dataset, class_names, device)`.

- [ ] **Step 1: Write failing transform and class-order tests**

```python
def test_eval_transform_is_deterministic(model_config, rgb_image):
    transform = build_transforms(model_config, train=False)
    torch.testing.assert_close(transform(rgb_image), transform(rgb_image))
    assert transform(rgb_image).shape == (3, 224, 224)


def test_imagefolder_order_must_match_checkpoint(ten_class_tree, model_config):
    datasets = create_datasets(ten_class_tree, model_config)
    assert tuple(datasets["train"].classes) == CLASS_NAMES
    shutil.rmtree(ten_class_tree / "val" / "trash")
    with pytest.raises(PipelineError, match="trash"):
        create_datasets(ten_class_tree, model_config)


def test_weights_follow_class_names_not_mapping_order(train_dataset):
    requested_order = tuple(reversed(CLASS_NAMES))
    tensor = ordered_class_weight_tensor(
        train_dataset, requested_order, torch.device("cpu")
    )
    assert tensor.shape == (10,)
    trash_index = train_dataset.class_to_idx["trash"]
    trash_count = train_dataset.targets.count(trash_index)
    expected = len(train_dataset) / (len(requested_order) * trash_count)
    assert tensor[0].item() == pytest.approx(expected)
```

- [ ] **Step 2: Run the tests and verify missing functions**

Run: `pytest tests/training/test_data.py -v`

Expected: FAIL because `src.training.data` is missing.

- [ ] **Step 3: Implement train and evaluation transforms**

```python
def build_transforms(config: Mapping[str, object], train: bool) -> transforms.Compose:
    size = int(config["input_size"])
    mean = config["normalization"]["mean"]
    std = config["normalization"]["std"]
    operations: list[Callable] = [transforms.Resize((size, size))]
    if train:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        ])
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(operations)
```

This is augmentation group C and is never used for validation/test. `create_datasets` instantiates `ImageFolder` for `train`, `val`, and `test`, verifies each `.classes` tuple exactly equals `CLASS_NAMES`, and reports the offending split path and missing/extra classes.

- [ ] **Step 4: Implement ordered weights and deterministic loaders**

```python
def ordered_class_weight_tensor(dataset: ImageFolder, class_names: Sequence[str],
                                device: torch.device) -> torch.Tensor:
    counts = Counter(int(target) for target in dataset.targets)
    total = len(dataset.targets)
    values = [
        total / (len(class_names) * counts[dataset.class_to_idx[class_name]])
        for class_name in class_names
    ]
    return torch.tensor(values, dtype=torch.float32, device=device)
```

`create_dataloaders` uses a seeded `torch.Generator`, shuffles only train, pins memory only for CUDA, and exposes batch-size/worker overrides. The training CLI invokes the post-split validation gate before calling it.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/training/test_data.py tests/preprocessing/test_dataloader.py -v`

Expected: PASS.

```powershell
git add src/training/data.py src/preprocessing/dataloader.py tests/training/test_data.py tests/preprocessing/test_dataloader.py
git commit -m "feat(training): add canonical loaders and ordered weights"
```

### Task 4: Two-phase training engine, resume, and checkpoint selection

**Files:**
- Create: `src/training/engine.py`
- Create: `src/training/train.py`
- Test: `tests/training/test_engine.py`
- Test: `tests/training/test_train_smoke.py`

**Interfaces:**
- Consumes: model factory, validated dataloaders, checkpoint envelope, dataset fingerprint, and model config.
- Produces: `configure_phase`, `run_epoch`, `fit`, `TrainingResult`, and CLI `python -m src.training.train`.

- [ ] **Step 1: Write failing phase and single-step tests**

```python
def test_phase_one_freezes_features_and_phase_two_unfreezes(model):
    configure_phase(model, phase=1)
    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    configure_phase(model, phase=2)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_one_cpu_optimizer_step_returns_finite_metrics(tiny_model, synthetic_loader):
    result = run_epoch(
        tiny_model, synthetic_loader,
        torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        torch.optim.AdamW(tiny_model.parameters(), lr=1e-3),
        torch.device("cpu"), scaler=None, gradient_clip=1.0,
    )
    assert math.isfinite(result.loss)
    assert 0.0 <= result.macro_f1 <= 1.0
```

- [ ] **Step 2: Run the engine tests and verify failure**

Run: `pytest tests/training/test_engine.py -v`

Expected: FAIL because `src.training.engine` is missing.

- [ ] **Step 3: Implement phase setup and epoch execution**

```python
def configure_phase(model: nn.Module, phase: int) -> None:
    if phase not in (1, 2):
        raise ValueError(f"phase must be 1 or 2, received {phase}")
    for parameter in model.features.parameters():
        parameter.requires_grad = phase == 2
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def optimizer_for_phase(model: nn.Module, phase: int, weight_decay: float) -> AdamW:
    if phase == 1:
        return AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=weight_decay)
    return AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ], weight_decay=weight_decay)
```

`run_epoch` enters train mode only when an optimizer is supplied, moves batches to device, uses `torch.autocast` and `GradScaler` only on CUDA, clips gradients before stepping, and calculates loss plus macro-F1 from accumulated integer predictions/targets. Catch CUDA OOM only to re-raise `RuntimeError(f"CUDA out of memory at batch size {batch_size}; rerun with --batch-size {max(1, batch_size // 2)}")` with the original exception chained.

- [ ] **Step 4: Write failing best/last/resume behavior tests**

```python
def test_best_uses_validation_macro_f1_and_last_contains_resume_state(training_fixture):
    result = fit(**training_fixture.arguments, scripted_val_f1=[0.40, 0.61, 0.55])
    best = load_checkpoint(result.best_path, torch.device("cpu"))
    last = load_checkpoint(result.last_path, torch.device("cpu"))
    assert best.metadata.metrics["val_macro_f1"] == pytest.approx(0.61)
    assert best.resume_state is None
    assert {"optimizer", "scheduler", "scaler", "next_epoch", "phase"} <= set(last.resume_state)
```

Run: `pytest tests/training/test_train_smoke.py -v`

Expected: FAIL because `fit`/resume behavior is absent.

- [ ] **Step 5: Implement training orchestration and resume**

`fit` runs phase 1 for exactly five epochs, then phase 2 for at most twenty-five epochs, creates a cosine scheduler per phase, selects best strictly on validation macro-F1, and stops phase 2 after seven non-improving epochs. `last.pt` is updated atomically after every epoch with optimizer/scheduler/scaler, `next_epoch`, phase, best score, and patience counter. Resume restores every available state and begins at `next_epoch`; it rejects a dataset-fingerprint or class-order mismatch.

The CLI is exactly:

```text
python -m src.training.train --data-root data/processed/v1 --manifest data/metadata/v1/split_manifest.csv --config configs/model_config.yaml --output-dir artifacts/runs/<run-id> [--resume artifacts/runs/<run-id>/last.pt] [--batch-size 16] [--deterministic]
```

It seeds Python, NumPy, and PyTorch with `42`, writes `resolved_config.yaml` and `history.csv`, and never reads the test loader during `fit`.

- [ ] **Step 6: Run training smoke tests and commit**

Run: `pytest tests/training/test_engine.py tests/training/test_train_smoke.py -v`

Expected: PASS on CPU synthetic tensors.

Run: `python -m src.training.train --help`

Expected: exit code `0` and the arguments above.

```powershell
git add src/training/engine.py src/training/train.py tests/training/test_engine.py tests/training/test_train_smoke.py
git commit -m "feat(training): add resumable two-phase fine-tuning"
```

### Task 5: Ten-class evaluator and structured artifacts

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/metrics.py`
- Create: `src/evaluation/evaluate.py`
- Test: `tests/evaluation/test_metrics.py`
- Test: `tests/evaluation/test_evaluate.py`

**Interfaces:**
- Consumes: validated test `ImageFolder`, `LoadedCheckpoint`, and `build_model_from_checkpoint`.
- Produces: `MetricBundle`, `compute_multiclass_metrics`, `evaluate_checkpoint`, and the documented JSON/CSV/PNG artifacts.

- [ ] **Step 1: Write failing metric tests, including missing-class AUC**

```python
def test_metrics_have_ten_by_ten_confusion_matrix():
    targets = np.arange(10)
    probabilities = np.eye(10) * 0.9 + 0.01
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    result = compute_multiclass_metrics(targets, probabilities, CLASS_NAMES)
    assert result.accuracy == 1.0
    assert result.confusion_matrix.shape == (10, 10)
    assert result.roc_auc_ovr_macro is not None


def test_missing_class_returns_null_auc_without_crashing():
    targets = np.arange(9)
    probabilities = np.eye(10)[:9]
    with pytest.warns(UserWarning, match="AUC"):
        result = compute_multiclass_metrics(targets, probabilities, CLASS_NAMES)
    assert result.roc_auc_ovr_macro is None
    assert result.confusion_matrix.shape == (10, 10)
```

- [ ] **Step 2: Run metric tests and verify failure**

Run: `pytest tests/evaluation/test_metrics.py -v`

Expected: FAIL because `src.evaluation.metrics` is missing.

- [ ] **Step 3: Implement multiclass metrics**

```python
@dataclass(frozen=True)
class MetricBundle:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: list[dict[str, float | int | str]]
    confusion_matrix: np.ndarray
    normalized_confusion_matrix: np.ndarray
    roc_auc_ovr_macro: float | None
```

Use `argmax` predictions; `accuracy_score`; `precision_recall_fscore_support(labels=range(10), zero_division=0)`; `confusion_matrix(labels=range(10))`; safe row normalization with zero rows left at zero; and `roc_auc_score(targets, probabilities, labels=range(10), multi_class="ovr", average="macro")` only when every class index appears in targets. Build one per-class row for every checkpoint class regardless of support.

- [ ] **Step 4: Write the failing artifact integration test**

```python
def test_evaluator_writes_complete_artifact_set(checkpoint, ten_class_test_tree, tmp_path):
    result = evaluate_checkpoint(checkpoint, ten_class_test_tree, tmp_path / "evaluation", device="cpu")
    assert set(path.name for path in result.output_dir.iterdir()) == {
        "metrics.json", "per_class_metrics.csv", "confusion_matrix_raw.png",
        "confusion_matrix_normalized.png",
    }
    assert json.loads((result.output_dir / "metrics.json").read_text())["class_names"] == list(CLASS_NAMES)
```

Run: `pytest tests/evaluation/test_evaluate.py -v`

Expected: FAIL because `evaluate_checkpoint` is missing.

- [ ] **Step 5: Implement evaluator and CLI**

Load only one completed `best.pt`, build its model and transform from metadata, verify the test folder class order equals metadata, collect softmax probabilities under `torch.inference_mode()`, calculate metrics, and write the exact four artifacts. JSON uses `null` for unavailable AUC. Both plot axes use checkpoint `class_names`, raw values are integers, and normalized values are percentages.

CLI:

```text
python -m src.evaluation.evaluate --checkpoint artifacts/runs/<run-id>/best.pt --test-root data/processed/v1/test --output-dir outputs/evaluation/<run-id> --device auto
```

- [ ] **Step 6: Verify and commit evaluation**

Run: `pytest tests/evaluation -v`

Expected: PASS and the missing-class fixture emits exactly one warning.

```powershell
git add src/evaluation tests/evaluation
git commit -m "feat(evaluation): add robust ten-class reports"
```

### Task 6: Lazy metadata-driven predictor and JSON CLI

**Files:**
- Create: `src/inference/__init__.py`
- Create: `src/inference/predict.py`
- Test: `tests/inference/test_predict.py`
- Test: `tests/inference/test_predict_cli.py`

**Interfaces:**
- Consumes: checkpoint envelope/model reconstruction and metadata transform contract.
- Produces: `ScoredClass`, `Prediction`, `WastePredictor`, and `python -m src.inference.predict`.

- [ ] **Step 1: Write failing lazy-load/top-k tests**

```python
def test_predictor_is_lazy_and_maps_logits_through_checkpoint_metadata(monkeypatch, checkpoint, image):
    calls = []
    monkeypatch.setattr("src.inference.predict.load_checkpoint", lambda path, device: calls.append(path) or checkpoint)
    predictor = WastePredictor(Path("model.pt"), device="cpu", confidence_threshold=0.55)
    assert calls == []
    result = predictor.predict_pil(image, top_k=3)
    assert calls == [Path("model.pt")]
    assert result.top1.class_id == checkpoint.metadata.class_names[result.top1.index]
    assert len(result.topk) == 3


def test_low_confidence_does_not_replace_top_one(predictor, ambiguous_image):
    result = predictor.predict_pil(ambiguous_image, top_k=3)
    assert result.low_confidence is True
    assert result.top1 == result.topk[0]
```

- [ ] **Step 2: Run predictor tests and verify failure**

Run: `pytest tests/inference/test_predict.py -v`

Expected: FAIL because `src.inference.predict` is missing.

- [ ] **Step 3: Implement immutable prediction types and lazy loading**

```python
@dataclass(frozen=True)
class ScoredClass:
    index: int
    class_id: str
    probability: float


@dataclass(frozen=True)
class Prediction:
    top1: ScoredClass
    topk: tuple[ScoredClass, ...]
    low_confidence: bool


class WastePredictor:
    def __init__(self, checkpoint_path: Path, device: str = "auto",
                 confidence_threshold: float = 0.55):
        self.checkpoint_path = checkpoint_path
        self.device = resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self._loaded = None
        self._model = None
        self._transform = None
```

On the first prediction, load and validate one checkpoint, reconstruct one model, and build one transform from its input size and normalization. Convert every Pillow image to RGB, run with `torch.inference_mode()`, softmax once, call `torch.topk(k=min(top_k, num_classes))`, and map each index only through `loaded.metadata.class_names`. Reject `top_k < 1`, missing/corrupt file paths, and confidence thresholds outside `[0, 1]` with path/value-specific errors.

- [ ] **Step 4: Implement and test JSON CLI output**

```python
def prediction_to_dict(prediction: Prediction) -> dict[str, object]:
    return {
        "top1": asdict(prediction.top1),
        "top3": [asdict(item) for item in prediction.topk],
        "low_confidence": prediction.low_confidence,
    }
```

CLI:

```text
python -m src.inference.predict --checkpoint artifacts/runs/<run-id>/best.pt --image sample.jpg --top-k 3 --confidence-threshold 0.55 --device auto
```

Run: `pytest tests/inference -v`

Expected: PASS and stdout is one valid UTF-8 JSON object without log text.

- [ ] **Step 5: Run the core acceptance suite and commit**

Run: `pytest tests/models tests/training tests/evaluation tests/inference -q`

Expected: all tests PASS on CPU.

Run: `python -m compileall src`

Expected: exit code `0`.

```powershell
git add src/inference tests/inference
git commit -m "feat(inference): add metadata-safe top-k prediction"
```

## Plan Acceptance

- Task 1 proves `[2, 3, 224, 224] → [2, 10]` and keeps architecture construction in one module.
- Tasks 2–4 implement the shared class-order contract, dataset fingerprint, two checkpoint modes, resume state, approved optimizer schedule, validation-only selection, and CPU training smoke test.
- Task 5 generates accuracy, macro/weighted F1, every per-class metric, raw/normalized 10×10 confusion matrices, and safe one-vs-rest macro AUC behavior.
- Task 6 proves labels are resolved from checkpoint metadata, supplies top-1/top-3 and the configurable `0.55` warning without secretly changing top-1.
- The final delivery plan may import `WastePredictor` and `Prediction`; it must not rebuild a model or transform in UI code.
