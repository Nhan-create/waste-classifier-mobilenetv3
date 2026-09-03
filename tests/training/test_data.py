from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.schema import CLASS_NAMES, PipelineError
from src.training.data import (
    build_transforms,
    create_dataloaders,
    create_datasets,
    load_model_config,
    ordered_class_weight_tensor,
)


def create_dataset_tree(root: Path, train_battery_count: int = 1) -> None:
    for split_name in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            count = (
                train_battery_count
                if split_name == "train" and class_name == "battery"
                else 1
            )
            for index in range(count):
                path = root / split_name / class_name / f"{index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (24, 18), color=(index * 20, 40, 80)).save(path)


def test_eval_transform_is_deterministic_and_has_configured_shape() -> None:
    config = load_model_config(Path("configs/model_config.yaml"))
    transform = build_transforms(config, train=False)
    image = Image.new("RGB", (25, 31), "orange")

    first = transform(image)
    second = transform(image)

    assert first.shape == (3, 224, 224)
    torch.testing.assert_close(first, second)


def test_all_imagefolder_splits_use_canonical_class_order(tmp_path: Path) -> None:
    create_dataset_tree(tmp_path)
    config = load_model_config(Path("configs/model_config.yaml"))

    datasets = create_datasets(tmp_path, config)

    assert set(datasets) == {"train", "val", "test"}
    assert all(tuple(dataset.classes) == CLASS_NAMES for dataset in datasets.values())

    missing_file = tmp_path / "val" / "trash" / "0.png"
    missing_file.unlink()
    missing_file.parent.rmdir()
    with pytest.raises(PipelineError, match="val.*trash"):
        create_datasets(tmp_path, config)


def test_class_weights_follow_requested_names_not_index_iteration(
    tmp_path: Path,
) -> None:
    create_dataset_tree(tmp_path, train_battery_count=2)
    config = load_model_config(Path("configs/model_config.yaml"))
    dataset = create_datasets(tmp_path, config)["train"]
    requested_order = tuple(reversed(CLASS_NAMES))

    weights = ordered_class_weight_tensor(
        dataset,
        requested_order,
        torch.device("cpu"),
    )

    assert weights.shape == (10,)
    assert weights[0].item() == pytest.approx(11 / 10)
    assert weights[-1].item() == pytest.approx(11 / 20)


def test_dataloaders_shuffle_only_training_data(tmp_path: Path) -> None:
    create_dataset_tree(tmp_path)
    config = load_model_config(Path("configs/model_config.yaml"))
    datasets = create_datasets(tmp_path, config)

    loaders = create_dataloaders(
        datasets,
        config,
        batch_size=4,
        num_workers=0,
        device=torch.device("cpu"),
    )

    assert type(loaders["train"].sampler).__name__ == "RandomSampler"
    assert type(loaders["val"].sampler).__name__ == "SequentialSampler"
    assert type(loaders["test"].sampler).__name__ == "SequentialSampler"
