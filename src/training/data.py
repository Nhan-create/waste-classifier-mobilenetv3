"""Validated ImageFolder datasets and training DataLoaders."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.data.dataset import discover_class_names
from src.data.schema import CLASS_NAMES, PipelineError
from src.preprocessing.augmentation import augmentation_group


def load_model_config(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as error:
        raise PipelineError(f"Model config not found: {path}") from error
    if not isinstance(config, dict):
        raise PipelineError(f"Model config must be a mapping: {path}")
    for section in ("model", "input", "training", "inference", "checkpoint"):
        if not isinstance(config.get(section), dict):
            raise PipelineError(f"Model config has no {section!r} mapping: {path}")
    return config


def build_transforms(
    config: Mapping[str, object],
    *,
    train: bool,
) -> transforms.Compose:
    input_config = config["input"]
    if not isinstance(input_config, Mapping):
        raise PipelineError("Model input config must be a mapping")
    image_size = int(input_config["size"])
    mean = [float(value) for value in input_config["mean"]]
    std = [float(value) for value in input_config["std"]]
    if train:
        return augmentation_group(
            str(input_config.get("augmentation_group", "C")),
            image_size,
            mean,
            std,
            train=True,
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def create_datasets(
    data_root: Path,
    config: Mapping[str, object],
) -> dict[str, ImageFolder]:
    datasets: dict[str, ImageFolder] = {}
    for split_name in ("train", "val", "test"):
        split_root = data_root / split_name
        discover_class_names(split_root)
        dataset = ImageFolder(
            split_root,
            transform=build_transforms(config, train=split_name == "train"),
        )
        if tuple(dataset.classes) != CLASS_NAMES:
            raise PipelineError(
                f"ImageFolder class order for {split_name} is "
                f"{tuple(dataset.classes)}, expected {CLASS_NAMES}"
            )
        datasets[split_name] = dataset
    return datasets


def create_dataloaders(
    datasets: Mapping[str, ImageFolder],
    config: Mapping[str, object],
    *,
    batch_size: int | None,
    num_workers: int | None,
    device: torch.device,
) -> dict[str, DataLoader]:
    training_config = config["training"]
    if not isinstance(training_config, Mapping):
        raise PipelineError("Model training config must be a mapping")
    resolved_batch_size = batch_size or int(training_config["batch_size"])
    resolved_workers = (
        int(training_config["num_workers"])
        if num_workers is None
        else num_workers
    )
    generator = torch.Generator().manual_seed(int(training_config["seed"]))
    return {
        split_name: DataLoader(
            dataset,
            batch_size=resolved_batch_size,
            shuffle=split_name == "train",
            num_workers=resolved_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=resolved_workers > 0,
            generator=generator if split_name == "train" else None,
        )
        for split_name, dataset in datasets.items()
    }


def ordered_class_weight_tensor(
    dataset: ImageFolder,
    class_names: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    if set(class_names) != set(dataset.classes):
        raise PipelineError(
            f"Requested classes {tuple(class_names)} do not match "
            f"dataset classes {tuple(dataset.classes)}"
        )
    counts = Counter(int(target) for target in dataset.targets)
    total = len(dataset.targets)
    values: list[float] = []
    for class_name in class_names:
        class_index = dataset.class_to_idx[class_name]
        count = counts[class_index]
        if count == 0:
            raise PipelineError(f"Training class {class_name!r} has no images")
        values.append(total / (len(class_names) * count))
    return torch.tensor(values, dtype=torch.float32, device=device)
