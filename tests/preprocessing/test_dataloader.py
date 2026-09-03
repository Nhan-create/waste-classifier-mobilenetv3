from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import RandomResizedCrop

from src.data.schema import CLASS_NAMES
from src.preprocessing.dataloader import get_transforms, make_imagefolder_dataloader


def create_tree(root: Path) -> None:
    for class_name in CLASS_NAMES:
        path = root / class_name / "one.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 18), "white").save(path)


def test_evaluation_transform_is_deterministic() -> None:
    image = Image.new("RGB", (25, 30), "purple")
    transform = get_transforms(
        224,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        train=False,
    )

    first = transform(image)
    second = transform(image)

    assert first.shape == (3, 224, 224)
    torch.testing.assert_close(first, second)


def test_group_c_is_used_only_for_training() -> None:
    train_transform = get_transforms(
        224,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        train=True,
        aug_group="C",
    )
    eval_transform = get_transforms(
        224,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        train=False,
        aug_group="C",
    )

    assert any(isinstance(operation, RandomResizedCrop) for operation in train_transform.transforms)
    assert not any(isinstance(operation, RandomResizedCrop) for operation in eval_transform.transforms)


def test_imagefolder_loader_returns_canonical_class_order(tmp_path: Path) -> None:
    create_tree(tmp_path)

    loader, class_names = make_imagefolder_dataloader(
        tmp_path,
        config_path=Path("configs/preprocessing_config.yaml"),
        train=False,
        batch_size=4,
        num_workers=0,
    )

    images, labels = next(iter(loader))
    assert class_names == CLASS_NAMES
    assert images.shape == (4, 3, 224, 224)
    assert labels.dtype == torch.int64
