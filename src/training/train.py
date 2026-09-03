"""Two-phase MobileNetV3 training and command-line orchestration."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.manifest import fingerprint_manifest, read_manifest
from src.data.schema import CLASS_NAMES, PipelineError
from src.data.validation import (
    require_valid_processed_dataset,
    validate_processed_dataset,
)
from src.models.mobilenetv3 import build_model
from src.training.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)
from src.training.data import (
    create_dataloaders,
    create_datasets,
    load_model_config,
    ordered_class_weight_tensor,
)
from src.training.engine import optimizer_for_phase, run_epoch


@dataclass(frozen=True)
class TrainingResult:
    best_path: Path
    last_path: Path
    history: tuple[dict[str, float | int], ...]
    best_macro_f1: float


def _phase_epochs(training_config: Mapping[str, object], phase: int) -> int:
    phase_config = training_config[f"phase{phase}"]
    if not isinstance(phase_config, Mapping):
        raise TypeError(f"phase{phase} configuration must be a mapping")
    epochs = int(phase_config["epochs"])
    if epochs < 1:
        raise ValueError(f"phase{phase} epochs must be positive")
    return epochs


def _resume_values(
    resume_state: Mapping[str, object],
) -> tuple[int, int, int, float, int, list[dict[str, float | int]]]:
    required = {
        "optimizer",
        "scheduler",
        "scaler",
        "next_epoch",
        "phase",
        "global_epoch",
        "best_macro_f1",
        "patience_counter",
        "history",
    }
    missing = required - set(resume_state)
    if missing:
        raise CheckpointCompatibilityError(
            f"Resume checkpoint is missing state keys: {sorted(missing)}"
        )
    return (
        int(resume_state["phase"]),
        int(resume_state["next_epoch"]),
        int(resume_state["global_epoch"]),
        float(resume_state["best_macro_f1"]),
        int(resume_state["patience_counter"]),
        list(resume_state["history"]),
    )


def fit(
    *,
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    class_weights: torch.Tensor,
    config: Mapping[str, object],
    base_metadata: CheckpointMetadata,
    output_dir: Path,
    device: torch.device,
    resume_path: Path | None = None,
) -> TrainingResult:
    """Train both phases, checkpointing strictly on validation macro-F1."""

    if set(loaders) < {"train", "val"}:
        raise ValueError("Training requires train and val DataLoaders")
    training_config = config["training"]
    if not isinstance(training_config, Mapping):
        raise TypeError("Training configuration must be a mapping")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=float(training_config["label_smoothing"]),
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    best_macro_f1 = float("-inf")
    patience_counter = 0
    global_epoch = 0
    history: list[dict[str, float | int]] = []
    resume_phase = 1
    resume_epoch = 0
    loaded_resume = None

    if resume_path is not None:
        loaded_resume = load_checkpoint(
            resume_path,
            device,
            expected_classes=base_metadata.class_names,
        )
        if loaded_resume.metadata.dataset_fingerprint != base_metadata.dataset_fingerprint:
            raise CheckpointCompatibilityError(
                "Resume checkpoint dataset fingerprint does not match current data"
            )
        if loaded_resume.resume_state is None:
            raise CheckpointCompatibilityError(
                "Resume checkpoint does not contain optimizer/scheduler state"
            )
        model.load_state_dict(loaded_resume.model_state, strict=True)
        (
            resume_phase,
            resume_epoch,
            global_epoch,
            best_macro_f1,
            patience_counter,
            history,
        ) = _resume_values(loaded_resume.resume_state)

    stop_training = False
    for phase in (1, 2):
        if phase < resume_phase:
            continue
        epochs = _phase_epochs(training_config, phase)
        start_epoch = resume_epoch if phase == resume_phase else 0
        if start_epoch >= epochs:
            continue
        optimizer = optimizer_for_phase(model, phase, training_config)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        if loaded_resume is not None and phase == resume_phase:
            resume_state = loaded_resume.resume_state
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            scaler_state = resume_state["scaler"]
            if scaler is not None and scaler_state is not None:
                scaler.load_state_dict(scaler_state)

        for phase_epoch in range(start_epoch, epochs):
            global_epoch += 1
            train_result = run_epoch(
                model=model,
                loader=loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                gradient_clip=float(training_config["gradient_clip"]),
                batch_size=int(training_config["batch_size"]),
            )
            val_result = run_epoch(
                model=model,
                loader=loaders["val"],
                criterion=criterion,
                optimizer=None,
                device=device,
                scaler=scaler,
                gradient_clip=float(training_config["gradient_clip"]),
                batch_size=int(training_config["batch_size"]),
            )
            scheduler.step()
            row: dict[str, float | int] = {
                "epoch": global_epoch,
                "phase": phase,
                "phase_epoch": phase_epoch,
                "train_loss": train_result.loss,
                "train_accuracy": train_result.accuracy,
                "train_macro_f1": train_result.macro_f1,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.accuracy,
                "val_macro_f1": val_result.macro_f1,
            }
            history.append(row)
            metadata = replace(
                base_metadata,
                epoch=global_epoch,
                metrics={
                    "train_loss": train_result.loss,
                    "train_accuracy": train_result.accuracy,
                    "train_macro_f1": train_result.macro_f1,
                    "val_loss": val_result.loss,
                    "val_accuracy": val_result.accuracy,
                    "val_macro_f1": val_result.macro_f1,
                },
            )

            if val_result.macro_f1 > best_macro_f1:
                best_macro_f1 = val_result.macro_f1
                patience_counter = 0
                save_checkpoint(best_path, metadata, model.state_dict())
            elif phase == 2:
                patience_counter += 1

            resume_state = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": None if scaler is None else scaler.state_dict(),
                "next_epoch": phase_epoch + 1,
                "phase": phase,
                "global_epoch": global_epoch,
                "best_macro_f1": best_macro_f1,
                "patience_counter": patience_counter,
                "history": history,
            }
            save_checkpoint(
                last_path,
                metadata,
                model.state_dict(),
                resume_state=resume_state,
            )
            if (
                phase == 2
                and patience_counter
                >= int(training_config["early_stopping_patience"])
            ):
                stop_training = True
                break
        loaded_resume = None
        resume_epoch = 0
        if stop_training:
            break

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("Training completed without producing best.pt and last.pt")
    return TrainingResult(
        best_path=best_path,
        last_path=last_path,
        history=tuple(history),
        best_macro_f1=best_macro_f1,
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device(requested)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = not deterministic


def _dataset_fingerprint(
    manifest_path: Path,
    preprocessing_config_path: Path,
) -> str:
    try:
        preprocessing = yaml.safe_load(
            preprocessing_config_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError as error:
        raise PipelineError(
            f"Preprocessing config not found: {preprocessing_config_path}"
        ) from error
    dataset_config = preprocessing["dataset"]
    mapping_path = Path(dataset_config["mapping_path"])
    split_config = dataset_config["split"]
    split_bytes = json.dumps(
        {
            "seed": int(split_config["seed"]),
            "train": float(split_config["train"]),
            "val": float(split_config["val"]),
            "test": float(split_config["test"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return fingerprint_manifest(
        read_manifest(manifest_path),
        mapping_path.read_bytes(),
        split_bytes,
    )


def _write_history(path: Path, history: tuple[dict[str, float | int], ...]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model_config.yaml"))
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=Path("configs/preprocessing_config.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        config = copy.deepcopy(load_model_config(arguments.config))
        training_config = config["training"]
        if arguments.batch_size is not None:
            if arguments.batch_size < 1:
                raise ValueError("--batch-size must be positive")
            training_config["batch_size"] = arguments.batch_size
        deterministic = bool(
            arguments.deterministic or training_config.get("deterministic", False)
        )
        seed = int(training_config["seed"])
        seed_everything(seed, deterministic)
        device = resolve_device(arguments.device)

        validation = validate_processed_dataset(
            arguments.data_root,
            arguments.manifest,
        )
        require_valid_processed_dataset(validation)
        fingerprint = _dataset_fingerprint(
            arguments.manifest,
            arguments.preprocessing_config,
        )
        datasets = create_datasets(arguments.data_root, config)
        loaders = create_dataloaders(
            datasets,
            config,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            device=device,
        )
        class_weights = ordered_class_weight_tensor(
            datasets["train"],
            CLASS_NAMES,
            device,
        )
        model_config = config["model"]
        input_config = config["input"]
        model = build_model(
            str(model_config["name"]),
            int(model_config["num_classes"]),
            pretrained=(
                bool(model_config["pretrained"])
                if arguments.resume is None
                else False
            ),
        )
        metadata = CheckpointMetadata(
            format_version=int(config["checkpoint"]["format_version"]),
            model_name=str(model_config["name"]),
            num_classes=int(model_config["num_classes"]),
            class_names=CLASS_NAMES,
            input_size=int(input_config["size"]),
            normalization={
                "mean": [float(value) for value in input_config["mean"]],
                "std": [float(value) for value in input_config["std"]],
            },
            epoch=0,
            metrics={},
            dataset_fingerprint=fingerprint,
            training_config=dict(training_config),
        )
        result = fit(
            model=model,
            loaders=loaders,
            class_weights=class_weights,
            config=config,
            base_metadata=metadata,
            output_dir=arguments.output_dir,
            device=device,
            resume_path=arguments.resume,
        )
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        (arguments.output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        _write_history(arguments.output_dir / "history.csv", result.history)
    except (
        CheckpointCompatibilityError,
        PipelineError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Training failed: {error}")
        return 2
    print(
        json.dumps(
            {
                "best_checkpoint": str(result.best_path),
                "last_checkpoint": str(result.last_path),
                "best_val_macro_f1": result.best_macro_f1,
                "epochs_completed": len(result.history),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
