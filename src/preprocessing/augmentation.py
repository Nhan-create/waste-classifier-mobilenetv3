from torchvision import transforms


def augmentation_group(
    group: str,
    image_size: int,
    mean: list[float],
    std: list[float],
    train: bool = True,
):
    # Define augmentation groups A (baseline) to E (strong)
    if not train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    if group == "A":
        # baseline
        tf = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    elif group == "B":
        tf = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    elif group == "C":
        tf = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    elif group == "D":
        tf = [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        # E: strongest (includes random erasing placeholder)
        tf = [
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ]

    return transforms.Compose(tf)
