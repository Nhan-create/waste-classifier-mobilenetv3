import os
from pathlib import Path
from typing import List, Tuple
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def load_config(cfg_path="configs/preprocessing_config.yaml"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def get_transforms(image_size: int, mean: List[float], std: List[float], train: bool = True, aug_group: str = None):
    if not train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    if aug_group is not None:
        try:
            from src.preprocessing.augmentation import augmentation_group
            return augmentation_group(aug_group, image_size, mean, std, train=True)
        except ImportError:
            try:
                from .augmentation import augmentation_group
                return augmentation_group(aug_group, image_size, mean, std, train=True)
            except ImportError:
                print("Warning: could not import augmentation_group, using default get_transforms.")

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class ImageFolderDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.transform:
                im = self.transform(im)
            else:
                im = transforms.ToTensor()(im)
        return im, label


def make_dataloader(items, batch_size=32, image_size=224, cfg_path="configs/preprocessing_config.yaml", train=True, aug_group=None, num_workers=4):
    cfg = load_config(cfg_path)
    mean = cfg.get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("std", [0.229, 0.224, 0.225])
    transform = get_transforms(image_size, mean, std, train=train, aug_group=aug_group)
    ds = ImageFolderDataset(items, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    return loader


if __name__ == "__main__":
    print("dataloader module — import in training scripts to build DataLoader.")
