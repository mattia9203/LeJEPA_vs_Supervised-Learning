"""ViT post-training extraction with patch tokens and GMAR maps.

This module is analysis-only: it loads fixed checkpoints and a fixed image
manifest, runs deterministic evaluation preprocessing, and writes artifacts
under outputs/analysis/gmar/vit/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from ..data.transforms import get_val_transforms
from ..network.model_factory import create_model


DEFAULT_SUPERVISED_CHECKPOINT = (
    "outputs/train/vit_supervised/ft_sup_last2_abcd/"
    "B_blr3em05_hlr0p0003_wd0p05/checkpoints/best.pt"
)
DEFAULT_LEJEPA_CHECKPOINT = (
    "outputs/train/vit_lejepa/ft_lejepa_regularized_from_probe/"
    "G_blr3em05_hlr0p0003_wd0p05/checkpoints/best.pt"
)
DEFAULT_MANIFEST = "outputs/analysis/manifests/analysis_val500_manifest.csv"
DEFAULT_OUTPUT_DIR = "outputs/analysis/gmar/vit"
SELECTED_BLOCKS = [3, 6, 9, 11]


@dataclass(frozen=True)
class ManifestItem:
    image_id: str
    image_path: Path
    true_label: int


class FixedManifestDataset(Dataset):
    """Dataset backed by the fixed analysis manifest."""

    def __init__(
        self,
        manifest_path: Path,
        transform: transforms.Compose,
        overlay_transform: transforms.Compose,
        max_images: int | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.transform = transform
        self.overlay_transform = overlay_transform
        self.items = self._read_manifest(manifest_path)
        if max_images is not None:
            self.items = self.items[:max_images]

    @staticmethod
    def _read_manifest(path: Path) -> List[ManifestItem]:
        items: List[ManifestItem] = []
        with open(path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                items.append(
                    ManifestItem(
                        image_id=f"image_{index:06d}",
                        image_path=Path(row["path"]),
                        true_label=int(row["label"]),
                    )
                )
        if not items:
            raise ValueError(f"No images found in manifest: {path}")
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        image = Image.open(item.image_path).convert("RGB")
        tensor = self.transform(image)
        overlay_image = self.overlay_transform(image)
        return {
            "image_id": item.image_id,
            "image_path": item.image_path.as_posix(),
            "true_label": item.true_label,
            "image": tensor,
            "overlay_image": np.asarray(overlay_image, dtype=np.uint8),
        }


def collate_manifest_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image_ids": [item["image_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "true_labels": torch.tensor(
            [item["true_label"] for item in batch],
            dtype=torch.long,
        ),
        "images": torch.stack([item["image"] for item in batch]),
        "overlay_images": [item["overlay_image"] for item in batch],
    }


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise KeyError(f"Checkpoint has no config: {checkpoint_path}")
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for module in model.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
    return model


def get_vit_blocks(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "blocks"):
        return backbone.blocks
    if backbone is not None and hasattr(backbone, "backbone"):
        nested_backbone = backbone.backbone
        if hasattr(nested_backbone, "blocks"):
            return nested_backbone.blocks
    raise TypeError(f"Could not find transformer blocks for {type(model)}")


def block_output_to_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor) and value.dim() == 3:
                return value
    if isinstance(output, dict):
        for value in output.values():
            if isinstance(value, torch.Tensor) and value.dim() == 3:
                return value
    raise TypeError(f"Cannot extract block tensor from output type {type(output)}")


def register_block_hooks(
    blocks: Sequence[torch.nn.Module],
    selected_blocks: Sequence[int],
    storage: Dict[int, torch.Tensor],
) -> List[Any]:
    handles = []
    selected_set = set(selected_blocks)

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            if index in selected_set:
                storage[index] = block_output_to_tensor(output).detach().cpu()

        return hook

    for index, block in enumerate(blocks):
        if index in selected_set:
            handles.append(block.register_forward_hook(make_hook(index)))
    return handles


def register_attention_hooks(
    blocks: Sequence[torch.nn.Module],
    storage: Dict[int, torch.Tensor],
) -> List[Any]:
    handles = []

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("Attention hook expected a tensor output.")
            if output.requires_grad:
                output.retain_grad()
            storage[index] = output

        return hook

    for index, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        attn_drop = getattr(attn, "attn_drop", None)
        if attn_drop is None:
            raise TypeError(f"Block {index} does not expose attn.attn_drop.")
        handles.append(attn_drop.register_forward_hook(make_hook(index)))
    return handles


def remove_hooks(handles: Iterable[Any]) -> None:
    for handle in handles:
        handle.remove()


def extract_patch_tokens(block_tensor: torch.Tensor, block_index: int) -> torch.Tensor:
    if block_tensor.dim() != 3:
        raise ValueError(
            f"Block {block_index} output must be [B, tokens, D], "
            f"got {tuple(block_tensor.shape)}"
        )
    patch_tokens = block_tensor[:, 1:, :]
    if patch_tokens.shape[1] != 196:
        raise ValueError(
            f"Block {block_index} expected 196 patch tokens after CLS, "
            f"got {patch_tokens.shape[1]} from {tuple(block_tensor.shape)}"
        )
    return patch_tokens


def save_tokens(
    output_dir: Path,
    model_name: str,
    block_index: int,
    image_ids: List[str],
    tokens: torch.Tensor,
    true_labels: List[int],
    predicted_labels: List[int],
) -> Dict[str, Any]:
    path = output_dir / "features" / model_name / "tokens" / f"block_{block_index:02d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "block_index": block_index,
        "image_ids": image_ids,
        "tokens": tokens,
        "tensor_shape": list(tokens.shape),
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
    }
    torch.save(payload, path)
    return {
        "path": path.as_posix(),
        "shape": list(tokens.shape),
        "has_nan": bool(torch.isnan(tokens).any().item()),
    }


def gmar_from_attentions(
    attentions: Dict[int, torch.Tensor],
    selected_block: int,
) -> np.ndarray:
    if selected_block not in attentions:
        raise KeyError(f"Missing attention for block {selected_block}")
    first = attentions[0]
    _, _, num_tokens, _ = first.shape
    rollout = torch.eye(num_tokens, device=first.device, dtype=first.dtype)

    for layer_index in range(selected_block + 1):
        attn = attentions[layer_index]
        grad = attn.grad
        if grad is None:
            raise RuntimeError(f"Missing attention gradient for block {layer_index}")
        attn = attn[0]
        grad = grad[0]
        head_scores = torch.relu(attn * grad).mean(dim=(-2, -1))
        head_scores = torch.nan_to_num(head_scores, nan=0.0, posinf=0.0, neginf=0.0)
        score_sum = head_scores.sum()
        if not torch.isfinite(score_sum) or score_sum <= 0:
            weights = torch.full_like(head_scores, 1.0 / head_scores.numel())
        else:
            weights = head_scores / score_sum
        fused = (weights[:, None, None] * attn).sum(dim=0)
        identity = torch.eye(fused.shape[-1], device=fused.device, dtype=fused.dtype)
        fused = fused + identity
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = fused @ rollout

    patch_scores = rollout[0, 1:]
    if patch_scores.numel() != 196:
        raise ValueError(f"Expected 196 CLS-to-patch values, got {patch_scores.numel()}")
    grid_size = int(math.sqrt(patch_scores.numel()))
    grid = patch_scores.reshape(1, 1, grid_size, grid_size)
    upsampled = F.interpolate(
        grid,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    upsampled = upsampled.float()
    upsampled = torch.nan_to_num(upsampled, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = upsampled.min()
    max_value = upsampled.max()
    if (max_value - min_value) > 1e-12:
        upsampled = (upsampled - min_value) / (max_value - min_value)
    else:
        upsampled = torch.zeros_like(upsampled)
    return upsampled.detach().cpu().numpy().astype(np.float32)


def save_overlay(
    base_image: np.ndarray,
    saliency: np.ndarray,
    output_path: Path,
    alpha: float = 0.45,
) -> None:
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
    overlay = (1.0 - alpha) * base + alpha * heatmap
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)


def extract_predictions_and_tokens(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    output_dir: Path,
    selected_blocks: Sequence[int],
    device: torch.device,
) -> Dict[str, Any]:
    blocks = get_vit_blocks(model)
    block_storage: Dict[int, torch.Tensor] = {}
    handles = register_block_hooks(blocks, selected_blocks, block_storage)
    image_ids: List[str] = []
    image_paths: List[str] = []
    true_labels: List[int] = []
    predicted_labels: List[int] = []
    confidences: List[float] = []
    token_parts: Dict[int, List[torch.Tensor]] = {block: [] for block in selected_blocks}

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{model_name} predictions/tokens"):
                block_storage.clear()
                images = batch["images"].to(device, non_blocking=True)
                logits = model(images)
                probabilities = logits.softmax(dim=-1)
                confidence, predictions = probabilities.max(dim=-1)

                image_ids.extend(batch["image_ids"])
                image_paths.extend(batch["image_paths"])
                true_labels.extend(batch["true_labels"].tolist())
                predicted_labels.extend(predictions.cpu().tolist())
                confidences.extend(confidence.cpu().tolist())

                for block_index in selected_blocks:
                    tokens = extract_patch_tokens(block_storage[block_index], block_index)
                    token_parts[block_index].append(tokens)
    finally:
        remove_hooks(handles)

    token_reports = {}
    for block_index in selected_blocks:
        tokens = torch.cat(token_parts[block_index], dim=0)
        token_reports[f"block_{block_index:02d}"] = save_tokens(
            output_dir,
            model_name,
            block_index,
            image_ids,
            tokens,
            true_labels,
            predicted_labels,
        )

    prediction_rows = []
    for index, image_id in enumerate(image_ids):
        prediction_rows.append(
            {
                "image_id": image_id,
                "image_path": image_paths[index],
                "true_label": true_labels[index],
                "model_name": model_name,
                "predicted_label": predicted_labels[index],
                "confidence": confidences[index],
                "correct": int(true_labels[index] == predicted_labels[index]),
            }
        )

    accuracy = 100.0 * sum(row["correct"] for row in prediction_rows) / len(prediction_rows)
    return {
        "rows": prediction_rows,
        "image_ids": image_ids,
        "predicted_labels": predicted_labels,
        "accuracy": accuracy,
        "token_reports": token_reports,
    }


def extract_gmar_maps(
    model: torch.nn.Module,
    model_name: str,
    dataset: FixedManifestDataset,
    predictions: Dict[str, int],
    output_dir: Path,
    selected_blocks: Sequence[int],
    device: torch.device,
) -> Dict[str, Any]:
    blocks = get_vit_blocks(model)
    attn_storage: Dict[int, torch.Tensor] = {}
    handles = register_attention_hooks(blocks, attn_storage)
    block_counts = {f"block_{block:02d}": 0 for block in selected_blocks}
    block_nan = {f"block_{block:02d}": False for block in selected_blocks}
    example_shape = None

    try:
        for item in tqdm(dataset, desc=f"{model_name} GMAR"):
            model.zero_grad(set_to_none=True)
            attn_storage.clear()
            image = item["image"].unsqueeze(0).to(device)
            image.requires_grad_(True)
            logits = model(image)
            target_class = predictions[item["image_id"]]
            target_score = logits[0, target_class]
            target_score.backward()

            for block_index in selected_blocks:
                saliency = gmar_from_attentions(attn_storage, block_index)
                block_name = f"block_{block_index:02d}"
                out_dir = output_dir / "saliency" / model_name / "gmar" / block_name
                out_dir.mkdir(parents=True, exist_ok=True)
                npy_path = out_dir / f"{item['image_id']}.npy"
                overlay_path = out_dir / f"{item['image_id']}_overlay.png"
                np.save(npy_path, saliency)
                save_overlay(item["overlay_image"], saliency, overlay_path)
                block_counts[block_name] += 1
                block_nan[block_name] = block_nan[block_name] or bool(np.isnan(saliency).any())
                example_shape = list(saliency.shape)

            model.zero_grad(set_to_none=True)
    finally:
        remove_hooks(handles)

    return {
        "block_counts": block_counts,
        "block_has_nan": block_nan,
        "example_shape": example_shape,
    }


def write_predictions_csv(output_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    path = output_dir / "metadata" / "vit_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "image_path",
        "true_label",
        "model_name",
        "predicted_label",
        "confidence",
        "correct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ViT tokens and GMAR maps.")
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
    manifest_path = Path(args.manifest)

    eval_transform = get_val_transforms(image_size=224, resize_size=256)
    overlay_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )
    dataset = FixedManifestDataset(
        manifest_path,
        transform=eval_transform,
        overlay_transform=overlay_transform,
        max_images=args.max_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_manifest_batch,
        pin_memory=device.type == "cuda",
    )

    model_specs = [
        ("vit_supervised", Path(args.supervised_checkpoint)),
        ("vit_lejepa", Path(args.lejepa_checkpoint)),
    ]
    all_prediction_rows: List[Dict[str, Any]] = []
    reports: Dict[str, Any] = {}
    image_ids_by_model: Dict[str, List[str]] = {}

    for model_name, checkpoint_path in model_specs:
        print(f"\nLoading {model_name}: {checkpoint_path}")
        model = load_checkpoint_model(checkpoint_path, device)
        token_prediction_report = extract_predictions_and_tokens(
            model,
            model_name,
            loader,
            output_dir,
            SELECTED_BLOCKS,
            device,
        )
        predictions_by_id = {
            row["image_id"]: int(row["predicted_label"])
            for row in token_prediction_report["rows"]
        }
        gmar_report = extract_gmar_maps(
            model,
            model_name,
            dataset,
            predictions_by_id,
            output_dir,
            SELECTED_BLOCKS,
            device,
        )
        all_prediction_rows.extend(token_prediction_report["rows"])
        image_ids_by_model[model_name] = token_prediction_report["image_ids"]
        reports[model_name] = {
            "num_images": len(token_prediction_report["rows"]),
            "accuracy": token_prediction_report["accuracy"],
            "tokens": token_prediction_report["token_reports"],
            "gmar": gmar_report,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions_path = write_predictions_csv(output_dir, all_prediction_rows)
    ids_match = image_ids_by_model["vit_supervised"] == image_ids_by_model["vit_lejepa"]
    reports["image_ids_identical"] = ids_match
    reports["predictions_csv"] = predictions_path.as_posix()

    report_path = output_dir / "metadata" / "vit_gmar_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2)

    print("\nSanity report")
    print(f"Predictions CSV: {predictions_path}")
    for model_name in ("vit_supervised", "vit_lejepa"):
        report = reports[model_name]
        print(f"\n{model_name}")
        print(f"  images processed: {report['num_images']}")
        print(f"  fixed-subset accuracy: {report['accuracy']:.2f}%")
        for block_name, token_report in report["tokens"].items():
            print(
                f"  tokens {block_name}: shape={token_report['shape']} "
                f"nan={token_report['has_nan']}"
            )
        print(f"  GMAR blocks: {report['gmar']['block_counts']}")
        print(f"  GMAR example shape: {report['gmar']['example_shape']}")
        print(f"  GMAR NaN checks: {report['gmar']['block_has_nan']}")
    print(f"\nimage_ids identical between models: {ids_match}")
    print(f"Report JSON: {report_path}")


if __name__ == "__main__":
    main()
