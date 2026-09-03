"""Versioned, self-describing MobileNetV3 checkpoint contract."""

from __future__ import annotations

import string
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from src.data.schema import CLASS_NAMES
from src.models.mobilenetv3 import MODEL_ID, build_model

FORMAT_VERSION = 1


class CheckpointCompatibilityError(RuntimeError):
    """A checkpoint cannot be interpreted without risking wrong labels."""


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


def _metadata_from_mapping(values: Mapping[str, object]) -> CheckpointMetadata:
    required = {field.name for field in CheckpointMetadata.__dataclass_fields__.values()}
    missing = required - set(values)
    extra = set(values) - required
    if missing or extra:
        raise CheckpointCompatibilityError(
            f"Invalid checkpoint metadata keys; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    try:
        return CheckpointMetadata(
            format_version=int(values["format_version"]),
            model_name=str(values["model_name"]),
            num_classes=int(values["num_classes"]),
            class_names=tuple(values["class_names"]),
            input_size=int(values["input_size"]),
            normalization={
                key: [float(item) for item in items]
                for key, items in dict(values["normalization"]).items()
            },
            epoch=int(values["epoch"]),
            metrics=dict(values["metrics"]),
            dataset_fingerprint=str(values["dataset_fingerprint"]),
            training_config=dict(values["training_config"]),
        )
    except (TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(
            f"Invalid checkpoint metadata value: {error}"
        ) from error


def _validate_metadata(
    metadata: CheckpointMetadata,
    expected_classes: Sequence[str] | None = None,
) -> None:
    if metadata.format_version != FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            f"Unsupported format_version {metadata.format_version}; "
            f"expected {FORMAT_VERSION}"
        )
    if metadata.model_name != MODEL_ID:
        raise CheckpointCompatibilityError(
            f"Unsupported model_name {metadata.model_name!r}; expected {MODEL_ID!r}"
        )
    if metadata.num_classes != len(metadata.class_names):
        raise CheckpointCompatibilityError(
            "num_classes does not match checkpoint class_names length"
        )
    if metadata.class_names != CLASS_NAMES:
        raise CheckpointCompatibilityError(
            f"Checkpoint class order must be canonical: {CLASS_NAMES}"
        )
    if len(set(metadata.class_names)) != metadata.num_classes:
        raise CheckpointCompatibilityError("Checkpoint class_names contain duplicates")
    if expected_classes is not None and tuple(expected_classes) != metadata.class_names:
        raise CheckpointCompatibilityError(
            f"Checkpoint class order {metadata.class_names} does not match "
            f"expected class order {tuple(expected_classes)}"
        )
    if metadata.input_size <= 0:
        raise CheckpointCompatibilityError("input_size must be positive")
    if set(metadata.normalization) != {"mean", "std"} or any(
        len(metadata.normalization[key]) != 3 for key in ("mean", "std")
    ):
        raise CheckpointCompatibilityError(
            "normalization must contain three-value mean and std"
        )
    fingerprint = metadata.dataset_fingerprint
    if len(fingerprint) != 64 or any(
        character not in string.hexdigits for character in fingerprint
    ):
        raise CheckpointCompatibilityError(
            "dataset_fingerprint must be a 64-character SHA-256 hex digest"
        )


def _validate_model_state(
    model_state: Mapping[str, object],
    num_classes: int,
) -> None:
    weight = model_state.get("classifier.3.weight")
    bias = model_state.get("classifier.3.bias")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise CheckpointCompatibilityError(
            "Checkpoint classifier.3 weight and bias tensors are required"
        )
    if weight.ndim != 2 or bias.ndim != 1:
        raise CheckpointCompatibilityError("Checkpoint classifier tensor rank is invalid")
    if weight.shape[0] != num_classes or bias.shape[0] != num_classes:
        raise CheckpointCompatibilityError(
            f"Checkpoint classifier head has {weight.shape[0]} outputs; "
            f"expected {num_classes}"
        )


def _validated_checkpoint(
    envelope: object,
    expected_classes: Sequence[str] | None = None,
) -> LoadedCheckpoint:
    if not isinstance(envelope, Mapping):
        raise CheckpointCompatibilityError("Checkpoint root must be a mapping")
    required = {"metadata", "model_state", "resume_state"}
    if set(envelope) != required:
        raise CheckpointCompatibilityError(
            f"Checkpoint keys must be {sorted(required)}, received {sorted(envelope)}"
        )
    metadata_values = envelope["metadata"]
    model_state_values = envelope["model_state"]
    resume_values = envelope["resume_state"]
    if not isinstance(metadata_values, Mapping):
        raise CheckpointCompatibilityError("Checkpoint metadata must be a mapping")
    if not isinstance(model_state_values, Mapping):
        raise CheckpointCompatibilityError("Checkpoint model_state must be a mapping")
    if resume_values is not None and not isinstance(resume_values, Mapping):
        raise CheckpointCompatibilityError("Checkpoint resume_state must be a mapping or null")
    metadata = _metadata_from_mapping(metadata_values)
    _validate_metadata(metadata, expected_classes)
    _validate_model_state(model_state_values, metadata.num_classes)
    model_state = {
        str(name): tensor for name, tensor in model_state_values.items()
    }
    return LoadedCheckpoint(
        metadata=metadata,
        model_state=model_state,
        resume_state=None if resume_values is None else dict(resume_values),
    )


def save_checkpoint(
    path: Path,
    metadata: CheckpointMetadata,
    model_state: Mapping[str, torch.Tensor],
    resume_state: Mapping[str, object] | None = None,
) -> None:
    envelope = {
        "metadata": asdict(metadata),
        "model_state": dict(model_state),
        "resume_state": None if resume_state is None else dict(resume_state),
    }
    _validated_checkpoint(envelope)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(envelope, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: Path,
    device: torch.device,
    expected_classes: Sequence[str] | None = None,
) -> LoadedCheckpoint:
    try:
        envelope = torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError as error:
        raise CheckpointCompatibilityError(f"Checkpoint not found: {path}") from error
    return _validated_checkpoint(envelope, expected_classes)


def build_model_from_checkpoint(
    loaded: LoadedCheckpoint,
    device: torch.device,
) -> nn.Module:
    model = build_model(
        loaded.metadata.model_name,
        loaded.metadata.num_classes,
        pretrained=False,
    )
    try:
        model.load_state_dict(loaded.model_state, strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(
            f"Checkpoint tensors do not match {loaded.metadata.model_name}: {error}"
        ) from error
    return model.to(device).eval()
