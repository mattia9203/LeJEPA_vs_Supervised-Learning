from torchvision import transforms

from ..globals import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(
    image_size: int = IMAGE_SIZE,
    color_jitter: bool = False,
    color_jitter_strength: float = 0.2,
    crop_scale: tuple[float, float] = (0.5, 1.0),
) -> transforms.Compose:
    transform_list = [
        transforms.RandomResizedCrop(image_size, scale=crop_scale),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=color_jitter_strength,
                contrast=color_jitter_strength,
                saturation=color_jitter_strength,
            )
        )
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(transform_list)


def get_val_transforms(
    image_size: int = IMAGE_SIZE,
    resize_size: int = 256,
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
