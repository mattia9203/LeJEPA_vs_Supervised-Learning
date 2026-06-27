from typing import Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..globals import IMAGENET_MEAN, IMAGENET_STD


def _multicrop_transform(
    size: int,
    scale: Sequence[float],
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=tuple(scale)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class MultiCropDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        num_global_views: int = 2,
        global_crop_size: int = 224,
        global_crop_scale: Sequence[float] = (0.3, 1.0),
        num_local_views: int = 6,
        local_crop_size: int = 98,
        local_crop_scale: Sequence[float] = (0.05, 0.3),
    ) -> None:
        if num_global_views < 2:
            raise ValueError("LeJEPA requires at least two global views.")
        if num_local_views < 0:
            raise ValueError("num_local_views cannot be negative.")
        self.dataset = dataset
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.global_transform = _multicrop_transform(
            global_crop_size,
            global_crop_scale,
        )
        self.local_transform = _multicrop_transform(
            local_crop_size,
            local_crop_scale,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image, label = self.dataset[index]
        if not isinstance(image, Image.Image):
            raise TypeError("MultiCropDataset expects the wrapped dataset to return PIL images.")
        return {
            "global_views": [
                self.global_transform(image) for _ in range(self.num_global_views)
            ],
            "local_views": [
                self.local_transform(image) for _ in range(self.num_local_views)
            ],
            "label": label,
            "index": index,
        }


def multicrop_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    num_global = len(batch[0]["global_views"])
    num_local = len(batch[0]["local_views"])
    return {
        "global_views": [
            torch.stack([item["global_views"][view] for item in batch])
            for view in range(num_global)
        ],
        "local_views": [
            torch.stack([item["local_views"][view] for item in batch])
            for view in range(num_local)
        ],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "indices": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }
