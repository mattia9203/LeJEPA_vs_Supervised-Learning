from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from ..data.transforms import get_val_transforms
from ..network.model_factory import create_model


DEFAULT_MANIFEST = "outputs/manifests/analysis_val500_manifest.csv"
DEFAULT_OUTPUT_DIR = "outputs/xai/cnn/gradcam"
DEFAULT_SUPERVISED_CHECKPOINT = (
    "outputs/train/cnn_supervised/resnet50_sup_lr5e2_wd1e4_crop03/"
    "checkpoints/best_val_acc.pt"
)
DEFAULT_LEJEPA_CHECKPOINT = (
    "outputs/train/cnn_lejepa/linear_probe_bestprobe_e60_adamw/"
    "checkpoints/best_val_acc.pt"
)
MODELS = ["cnn_supervised", "cnn_lejepa"]
LAYERS = ["layer2", "layer3", "layer4"]


class FixedManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        image_transform: transforms.Compose,
        overlay_transform: transforms.Compose,
        max_images: int | None = None,
    ) -> None:
        self.items = []
        with open(manifest_path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                self.items.append(
                    {
                        "image_id": f"image_{index:06d}",
                        "image_path": row["path"],
                        "true_label": int(row["label"]),
                    }
                )
        if max_images is not None:
            self.items = self.items[:max_images]
        if not self.items:
            raise ValueError(f"No images found in manifest: {manifest_path}")
        self.image_transform = image_transform
        self.overlay_transform = overlay_transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        image = Image.open(item["image_path"]).convert("RGB")
        return {
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "true_label": item["true_label"],
            "image": self.image_transform(image),
            "overlay_image": np.asarray(self.overlay_transform(image), dtype=np.uint8),
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image_ids": [item["image_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "true_labels": torch.tensor([item["true_label"] for item in batch], dtype=torch.long),
        "images": torch.stack([item["image"] for item in batch]),
        "overlay_images": [item["overlay_image"] for item in batch],
    }


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise KeyError(f"Checkpoint has no config: {checkpoint_path}")
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_target_layers(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise TypeError(f"Model has no backbone: {type(model)}")
    return {name: getattr(backbone, name) for name in LAYERS}


def remove_hooks(handles: Iterable[Any]) -> None:
    for handle in handles:
        handle.remove()


def register_activation_hooks(
    target_layers: Dict[str, torch.nn.Module],
    storage: Dict[str, torch.Tensor],
) -> List[Any]:
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            storage[name] = output.detach().cpu().to(torch.float16)

        return hook

    for name, layer in target_layers.items():
        handles.append(layer.register_forward_hook(make_hook(name)))
    return handles


def register_gradcam_hooks(
    target_layers: Dict[str, torch.nn.Module],
    storage: Dict[str, torch.Tensor],
) -> List[Any]:
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if not output.requires_grad:
                raise RuntimeError(f"{name} output does not require grad.")
            output.retain_grad()
            storage[name] = output

        return hook

    for name, layer in target_layers.items():
        handles.append(layer.register_forward_hook(make_hook(name)))
    return handles


def normalize_map(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = values.min()
    max_value = values.max()
    if (max_value - min_value) <= 1e-8:
        return torch.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def gradcam_map(activation: torch.Tensor) -> np.ndarray:
    gradient = activation.grad
    if gradient is None:
        raise RuntimeError("Missing activation gradient for Grad-CAM.")
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True).relu()
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = normalize_map(cam)
    return cam.detach().cpu().numpy().astype(np.float32)


def save_overlay(base_image: np.ndarray, saliency: np.ndarray, output_path: Path) -> None:
    values = np.clip(saliency, 0.0, 1.0)
    heatmap = np.stack(
        [
            np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0),
        ],
        axis=-1,
    )
    base = base_image.astype(np.float32) / 255.0
    overlay = (0.55 * base) + (0.45 * heatmap)
    Image.fromarray(np.clip(overlay * 255.0, 0, 255).astype(np.uint8)).save(output_path)


def extract_predictions_and_activations(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    target_layers = get_target_layers(model)
    activation_storage: Dict[str, torch.Tensor] = {}
    handles = register_activation_hooks(target_layers, activation_storage)

    image_ids: List[str] = []
    image_paths: List[str] = []
    true_labels: List[int] = []
    predicted_labels: List[int] = []
    confidences: List[float] = []
    activation_parts: Dict[str, List[torch.Tensor]] = {name: [] for name in LAYERS}

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{model_name} predictions/activations"):
                activation_storage.clear()
                images = batch["images"].to(device, non_blocking=True)
                logits = model(images)
                probabilities = logits.softmax(dim=-1)
                confidence, predictions = probabilities.max(dim=-1)

                image_ids.extend(batch["image_ids"])
                image_paths.extend(batch["image_paths"])
                true_labels.extend(batch["true_labels"].tolist())
                predicted_labels.extend(predictions.cpu().tolist())
                confidences.extend(confidence.cpu().tolist())

                for layer_name in LAYERS:
                    activation_parts[layer_name].append(activation_storage[layer_name])
    finally:
        remove_hooks(handles)

    activation_reports: Dict[str, Any] = {}
    for layer_name in LAYERS:
        activations = torch.cat(activation_parts[layer_name], dim=0)
        path = output_dir / "features" / model_name / "activations" / f"{layer_name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": model_name,
            "layer_name": layer_name,
            "image_ids": image_ids,
            "activations": activations,
            "tensor_shape": list(activations.shape),
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
        }
        torch.save(payload, path)
        activation_reports[layer_name] = {
            "path": path.as_posix(),
            "shape": list(activations.shape),
            "has_nan": bool(torch.isnan(activations.float()).any().item()),
        }

    rows = []
    for index, image_id in enumerate(image_ids):
        rows.append(
            {
                "image_id": image_id,
                "image_path": image_paths[index],
                "true_label": true_labels[index],
                "model_name": model_name,
                "predicted_label": predicted_labels[index],
                "confidence": confidences[index],
                "correct": int(predicted_labels[index] == true_labels[index]),
            }
        )
    accuracy = 100.0 * sum(row["correct"] for row in rows) / len(rows)
    return {
        "rows": rows,
        "image_ids": image_ids,
        "predictions_by_id": {
            row["image_id"]: int(row["predicted_label"]) for row in rows
        },
        "accuracy": accuracy,
        "activation_reports": activation_reports,
    }


def extract_gradcam(
    model: torch.nn.Module,
    model_name: str,
    dataset: FixedManifestDataset,
    predictions_by_id: Dict[str, int],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    target_layers = get_target_layers(model)
    gradcam_storage: Dict[str, torch.Tensor] = {}
    handles = register_gradcam_hooks(target_layers, gradcam_storage)
    counts = {name: 0 for name in LAYERS}
    nan_counts = {name: 0 for name in LAYERS}
    example_shapes: Dict[str, List[int] | None] = {name: None for name in LAYERS}

    try:
        for item in tqdm(dataset, desc=f"{model_name} Grad-CAM"):
            model.zero_grad(set_to_none=True)
            gradcam_storage.clear()
            image = item["image"].unsqueeze(0).to(device)
            image.requires_grad_(True)
            logits = model(image)
            target_class = predictions_by_id[item["image_id"]]
            logits[0, target_class].backward()

            for layer_name in LAYERS:
                cam = gradcam_map(gradcam_storage[layer_name])
                out_dir = output_dir / "saliency" / model_name / "gradcam" / layer_name
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / f"{item['image_id']}.npy", cam)
                save_overlay(item["overlay_image"], cam, out_dir / f"{item['image_id']}_overlay.png")
                counts[layer_name] += 1
                nan_counts[layer_name] += int(np.isnan(cam).any())
                example_shapes[layer_name] = list(cam.shape)

            model.zero_grad(set_to_none=True)
    finally:
        remove_hooks(handles)

    return {
        "counts": counts,
        "nan_counts": nan_counts,
        "example_shapes": example_shapes,
    }


def write_predictions(output_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    path = output_dir / "metadata" / "cnn_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id",
        "image_path",
        "true_label",
        "model_name",
        "predicted_label",
        "confidence",
        "correct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--supervised_checkpoint", default=DEFAULT_SUPERVISED_CHECKPOINT)
    parser.add_argument("--lejepa_checkpoint", default=DEFAULT_LEJEPA_CHECKPOINT)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    image_transform = get_val_transforms(image_size=224, resize_size=256)
    overlay_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    dataset = FixedManifestDataset(
        Path(args.manifest),
        image_transform=image_transform,
        overlay_transform=overlay_transform,
        max_images=args.max_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    expected_ids = [item["image_id"] for item in dataset.items]

    specs = [
        ("cnn_supervised", Path(args.supervised_checkpoint)),
        ("cnn_lejepa", Path(args.lejepa_checkpoint)),
    ]
    all_rows: List[Dict[str, Any]] = []
    reports: Dict[str, Any] = {}
    ids_by_model: Dict[str, List[str]] = {}

    for model_name, checkpoint_path in specs:
        print(f"\nLoading {model_name}: {checkpoint_path}")
        model = load_model(checkpoint_path, device)
        pred_report = extract_predictions_and_activations(
            model,
            model_name,
            loader,
            output_dir,
            device,
        )
        gradcam_report = extract_gradcam(
            model,
            model_name,
            dataset,
            pred_report["predictions_by_id"],
            output_dir,
            device,
        )
        all_rows.extend(pred_report["rows"])
        ids_by_model[model_name] = pred_report["image_ids"]
        reports[model_name] = {
            "num_images": len(pred_report["rows"]),
            "accuracy": pred_report["accuracy"],
            "activation_layers_saved": list(pred_report["activation_reports"].keys()),
            "activation_reports": pred_report["activation_reports"],
            "gradcam_counts": gradcam_report["counts"],
            "gradcam_example_shapes": gradcam_report["example_shapes"],
            "activation_nan_counts": {
                layer: int(report["has_nan"]) for layer, report in pred_report["activation_reports"].items()
            },
            "gradcam_nan_counts": gradcam_report["nan_counts"],
            "image_ids_match_manifest": pred_report["image_ids"] == expected_ids,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions_path = write_predictions(output_dir, all_rows)
    reports["image_ids_identical_between_models"] = (
        ids_by_model["cnn_supervised"] == ids_by_model["cnn_lejepa"]
    )
    reports["image_ids_identical_to_manifest"] = all(
        ids == expected_ids for ids in ids_by_model.values()
    )
    reports["predictions_csv"] = predictions_path.as_posix()
    reports["manifest"] = Path(args.manifest).as_posix()

    report_path = output_dir / "reports" / "cnn_step3_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2)

    print("\nCNN Step 3 sanity report")
    print(f"Predictions CSV: {predictions_path}")
    for model_name in MODELS:
        report = reports[model_name]
        print(f"\n{model_name}")
        print(f"  images processed: {report['num_images']}")
        print(f"  fixed-subset accuracy: {report['accuracy']:.2f}%")
        for layer_name in LAYERS:
            print(
                f"  {layer_name}: activations={report['activation_reports'][layer_name]['shape']} "
                f"act_nan={report['activation_nan_counts'][layer_name]} "
                f"gradcam={report['gradcam_counts'][layer_name]} "
                f"gradcam_shape={report['gradcam_example_shapes'][layer_name]} "
                f"gradcam_nan={report['gradcam_nan_counts'][layer_name]}"
            )
    print(f"\nimage_ids identical between models: {reports['image_ids_identical_between_models']}")
    print(f"image_ids identical to manifest: {reports['image_ids_identical_to_manifest']}")
    print(f"Report JSON: {report_path}")


if __name__ == "__main__":
    main()
