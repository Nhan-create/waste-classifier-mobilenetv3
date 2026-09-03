"""Small, testable primitives for two-phase fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float


def configure_phase(model: nn.Module, phase: int) -> None:
    if phase not in (1, 2):
        raise ValueError(f"phase must be 1 or 2, received {phase}")
    if not hasattr(model, "features") or not hasattr(model, "classifier"):
        raise TypeError("Training model must expose features and classifier modules")
    for parameter in model.features.parameters():
        parameter.requires_grad = phase == 2
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def optimizer_for_phase(
    model: nn.Module,
    phase: int,
    training_config: Mapping[str, object],
) -> AdamW:
    configure_phase(model, phase)
    weight_decay = float(training_config["weight_decay"])
    if phase == 1:
        phase_config = training_config["phase1"]
        return AdamW(
            model.classifier.parameters(),
            lr=float(phase_config["head_lr"]),
            weight_decay=weight_decay,
        )
    phase_config = training_config["phase2"]
    return AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": float(phase_config["backbone_lr"]),
            },
            {
                "params": model.classifier.parameters(),
                "lr": float(phase_config["head_lr"]),
            },
        ],
        weight_decay=weight_decay,
    )


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    gradient_clip: float,
    batch_size: int,
) -> EpochResult:
    """Run one training or validation epoch and return aggregate metrics."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    try:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type=device.type,
                    enabled=scaler is not None,
                ):
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                if optimizer is not None:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                        optimizer.step()
            predictions = logits.argmax(dim=1)
            current_batch = targets.shape[0]
            total_examples += current_batch
            total_loss += float(loss.detach().item()) * current_batch
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
    except torch.cuda.OutOfMemoryError as error:
        suggested = max(1, batch_size // 2)
        raise RuntimeError(
            f"CUDA out of memory at batch size {batch_size}; "
            f"rerun with --batch-size {suggested}"
        ) from error

    if total_examples == 0:
        raise ValueError("DataLoader produced no examples")
    correct = sum(
        prediction == target
        for prediction, target in zip(all_predictions, all_targets, strict=True)
    )
    return EpochResult(
        loss=total_loss / total_examples,
        accuracy=correct / total_examples,
        macro_f1=float(
            f1_score(
                all_targets,
                all_predictions,
                average="macro",
                zero_division=0,
            )
        ),
    )
