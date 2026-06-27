from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .multicrop import MultiCropDataset, multicrop_collate
from .transforms import get_train_transforms, get_val_transforms


IMG_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


class ImageNet100Subset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        class_to_idx: Dict[str, int],
        split_dirs: List[Path],
        transform: Optional[Callable] = None,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.root = Path(root)
        self.split = split
        self.split_dirs = split_dirs
        self.class_to_idx = class_to_idx
        self.classes = list(class_to_idx.keys())
        self.transform = transform
        self.samples = self._load_samples()

        if not self.samples:
            raise RuntimeError(
                f"No images found for split '{split}' in {self.split_dirs}. "
                "Check that ImageNet-100 is arranged as split/<class_folder>/image.JPEG."
            )

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for class_name, label in self.class_to_idx.items():
            class_dirs = [split_dir / class_name for split_dir in self.split_dirs]
            existing_class_dirs = [class_dir for class_dir in class_dirs if class_dir.is_dir()]
            if not existing_class_dirs:
                raise FileNotFoundError(
                    f"Selected class '{class_name}' is missing from {self.split_dirs}."
                )

            for class_dir in existing_class_dirs:
                for image_path in sorted(class_dir.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in IMG_EXTENSIONS:
                        samples.append((str(image_path), label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def resolve_split_dirs(data_root: Path, split: str) -> List[Path]:
    if split == "train":
        shard_dirs = sorted(path for path in data_root.iterdir() if path.is_dir() and path.name.startswith("train.X"))
        if shard_dirs:
            return shard_dirs
    elif split == "val":
        val_x_dir = data_root / "val.X"
        if val_x_dir.is_dir():
            return [val_x_dir]

    raise FileNotFoundError(
        f"Could not find split '{split}' under {data_root}. "
        "Expected train.X* directories and val.X."
    )


def list_class_folders(split_dirs: List[Path]) -> List[str]:
    class_names = set()
    for split_dir in split_dirs:
        class_names.update(path.name for path in split_dir.iterdir() if path.is_dir())
    return sorted(class_names)


def discover_selected_classes(config: dict) -> Tuple[List[str], List[Path], List[Path]]:
    data_root = Path(config["data_root"])
    train_dirs = resolve_split_dirs(data_root, "train")
    val_dirs = resolve_split_dirs(data_root, "val")

    train_classes = list_class_folders(train_dirs)
    val_classes = set(list_class_folders(val_dirs))

    selected_classes = config.get("selected_classes")
    expected_count = int(config.get("num_classes", config.get("num_selected_classes", 30)))

    if selected_classes:
        selected = list(selected_classes)
    else:
        if len(train_classes) < expected_count:
            raise RuntimeError(
                f"Found only {len(train_classes)} train class folders, "
                f"but {expected_count} are required."
            )
        selected = train_classes[:expected_count]

    if len(selected) != expected_count:
        raise ValueError(
            f"Selected {len(selected)} classes, but num_classes is {expected_count}. "
            "Update selected_classes or num_classes in the config."
        )

    missing_in_train = [name for name in selected if name not in train_classes]
    missing_in_val = [name for name in selected if name not in val_classes]
    if missing_in_train:
        raise FileNotFoundError(f"Selected classes missing from train/: {missing_in_train}")
    if missing_in_val:
        raise FileNotFoundError(f"Selected classes missing from val/: {missing_in_val}")

    return selected, train_dirs, val_dirs


def get_imagenet100_loaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    data_root = config["data_root"]
    image_size = config.get("image_size", 224)
    val_resize_size = config.get("val_resize_size", 256)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)

    selected_classes, train_dirs, val_dirs = discover_selected_classes(config)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(selected_classes)}

    train_transform = get_train_transforms(
        image_size=image_size,
        color_jitter=config.get("train_color_jitter", False),
        color_jitter_strength=config.get("color_jitter_strength", 0.2),
        crop_scale=tuple(config.get("train_crop_scale", (0.5, 1.0))),
    )
    val_transform = get_val_transforms(
        image_size=image_size,
        resize_size=val_resize_size,
    )

    train_dataset = ImageNet100Subset(
        root=data_root,
        split="train",
        class_to_idx=class_to_idx,
        split_dirs=train_dirs,
        transform=train_transform,
    )
    val_dataset = ImageNet100Subset(
        root=data_root,
        split="val",
        class_to_idx=class_to_idx,
        split_dirs=val_dirs,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_imagenet100_multicrop_loader(
    config: dict,
) -> Tuple[DataLoader, List[str], Dict[str, int]]:
    data_root = config["data_root"]
    selected_classes, train_dirs, _ = discover_selected_classes(config)
    class_to_idx = {
        class_name: index for index, class_name in enumerate(selected_classes)
    }
    base_dataset = ImageNet100Subset(
        root=data_root,
        split="train",
        class_to_idx=class_to_idx,
        split_dirs=train_dirs,
        transform=None,
    )
    dataset = MultiCropDataset(
        base_dataset,
        num_global_views=config.get("num_global_views", 2),
        global_crop_size=config.get("global_crop_size", 224),
        global_crop_scale=config.get("global_crop_scale", (0.3, 1.0)),
        num_local_views=config.get("num_local_views", 6),
        local_crop_size=config.get("local_crop_size", 98),
        local_crop_scale=config.get("local_crop_scale", (0.05, 0.3)),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.get("micro_batch_size", 16),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
        collate_fn=multicrop_collate,
    )
    return loader, selected_classes, class_to_idx


def get_imagenet100_probe_loaders(
    config: dict,
    train_fraction_per_class: float | None = None,
) -> Tuple[DataLoader, DataLoader]:
    import random
    from torch.utils.data import Subset

    selected_classes, train_dirs, val_dirs = discover_selected_classes(config)
    class_to_idx = {
        class_name: index for index, class_name in enumerate(selected_classes)
    }
    transform = get_val_transforms(
        image_size=config.get("image_size", 224),
        resize_size=config.get("val_resize_size", 256),
    )
    train_dataset = ImageNet100Subset(
        root=config["data_root"],
        split="train",
        class_to_idx=class_to_idx,
        split_dirs=train_dirs,
        transform=transform,
    )
    val_dataset = ImageNet100Subset(
        root=config["data_root"],
        split="val",
        class_to_idx=class_to_idx,
        split_dirs=val_dirs,
        transform=transform,
    )

    if train_fraction_per_class is not None and train_fraction_per_class < 1.0:
        rng = random.Random(config.get("probe_seed", 42))
        per_class: Dict[int, List[int]] = {
            label: [] for label in range(len(selected_classes))
        }
        for index, (_, label) in enumerate(train_dataset.samples):
            per_class[label].append(index)
        selected_indices = []
        for indices in per_class.values():
            count = max(1, round(len(indices) * train_fraction_per_class))
            selected_indices.extend(rng.sample(indices, count))
        train_dataset = Subset(train_dataset, sorted(selected_indices))

    batch_size = config.get("probe_batch_size", 256)
    workers = config.get("num_workers", 4)
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        ),
    )
