"""Canonical ImageFolder transforms and DataLoader construction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.data.dataset import discover_class_names
from src.data.schema import CLASS_NAMES, PipelineError
from src.preprocessing.augmentation import augmentation_group


def load_config(config_path: Path | str = "configs/preprocessing_config.yaml") -> dict:
    path = Path(config_path)
    try:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as error:
        raise PipelineError(f"Preprocessing config not found: {path}") from error
    if not isinstance(config, dict):
        raise PipelineError(f"Preprocessing config must be a mapping: {path}")
    return config


def get_transforms(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    train: bool = True,
    aug_group: str | None = None,
) -> transforms.Compose:
    if train:
        return augmentation_group(
            aug_group or "C",
            image_size,
            list(mean),
            list(std),
            train=True,
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def make_imagefolder_dataloader(
    split_dir: Path,
    *,
    config_path: Path = Path("configs/preprocessing_config.yaml"),
    train: bool,
    class_names: Sequence[str] = CLASS_NAMES,
    batch_size: int | None = None,
    num_workers: int | None = None,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, tuple[str, ...]]:
    discovered = discover_class_names(split_dir)
    if tuple(class_names) != discovered:
        raise PipelineError(
            f"Requested class order {tuple(class_names)} does not match {discovered}"
        )
    config = load_config(config_path)
    transform = get_transforms(
        int(config.get("image_size", 224)),
        config.get("mean", [0.485, 0.456, 0.406]),
        config.get("std", [0.229, 0.224, 0.225]),
        train=train,
        aug_group="C" if train else None,
    )
    dataset = ImageFolder(split_dir, transform=transform)
    if tuple(dataset.classes) != tuple(class_names):
        raise PipelineError(
            f"ImageFolder class order {tuple(dataset.classes)} does not match "
            f"{tuple(class_names)}"
        )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or int(config.get("batch_size", 32)),
        shuffle=train,
        num_workers=(
            int(config.get("num_workers", 4))
            if num_workers is None
            else num_workers
        ),
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        generator=generator,
    )
    return loader, tuple(dataset.classes)
