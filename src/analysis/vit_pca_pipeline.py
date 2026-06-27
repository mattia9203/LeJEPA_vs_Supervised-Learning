"""Generate ViT PCA maps from patch tokens extracted in the GMAR step."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


DEFAULT_TOKENS_ROOT = "outputs/analysis/gmar/vit/features"
DEFAULT_MANIFEST = "outputs/analysis/manifests/analysis_val500_manifest.csv"
DEFAULT_OUTPUT_DIR = "outputs/analysis/pca/vit"
MODELS = ["vit_supervised", "vit_lejepa"]
BLOCKS = ["block_03", "block_06", "block_09", "block_11"]


def read_manifest(manifest_path: Path) -> Dict[str, Path]:
    """Map fixed image ids to original image paths using manifest order."""
    mapping: Dict[str, Path] = {}
    with open(manifest_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            mapping[f"image_{index:06d}"] = Path(row["path"])
    if not mapping:
        raise ValueError(f"No rows found in manifest: {manifest_path}")
    return mapping


def load_overlay_image(path: Path) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )
    image = Image.open(path).convert("RGB")
    return np.asarray(transform(image), dtype=np.uint8)


def normalize_map(values: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, bool]:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = values.min()
    max_value = values.max()
    if (max_value - min_value) <= eps:
        return torch.zeros_like(values), True
    return (values - min_value) / (max_value - min_value), False


def compute_pca_map(tokens: torch.Tensor, eps: float = 1e-8) -> tuple[np.ndarray, bool]:
    """Compute abs(PC1) spatial map from one image's [196, D] patch tokens."""
    if tokens.dim() != 2:
        raise ValueError(f"Expected tokens [196, D], got {tuple(tokens.shape)}")
    if tokens.shape[0] != 196:
        raise ValueError(f"Expected 196 patch tokens, got {tokens.shape[0]}")

    tokens = torch.nan_to_num(tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    if centered.norm() <= eps:
        return np.zeros((224, 224), dtype=np.float32), True

    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        pc1 = centered @ vh[0]
    except RuntimeError:
        # Fallback for rare SVD convergence issues on nearly degenerate data.
        _, _, vh = torch.pca_lowrank(centered, q=1, center=False)
        pc1 = centered @ vh[:, 0]

    pc1 = pc1.abs()
    grid_size = int(math.sqrt(pc1.numel()))
    if grid_size * grid_size != pc1.numel():
        raise ValueError(f"Cannot reshape {pc1.numel()} PCA values to a square grid.")
    grid = pc1.reshape(1, 1, grid_size, grid_size)
    upsampled = F.interpolate(
        grid,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    normalized, constant = normalize_map(upsampled, eps=eps)
    return normalized.cpu().numpy().astype(np.float32), constant


def top20_mask(pca_map: np.ndarray, constant: bool) -> np.ndarray:
    if constant:
        return np.zeros_like(pca_map, dtype=bool)
    flat = pca_map.reshape(-1)
    count = max(1, int(math.ceil(flat.size * 0.20)))
    selected = np.argpartition(flat, -count)[-count:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(pca_map.shape)


def otsu_mask(pca_map: np.ndarray, constant: bool) -> np.ndarray:
    if constant:
        return np.zeros_like(pca_map, dtype=bool)
    hist, bin_edges = np.histogram(pca_map.reshape(-1), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return np.zeros_like(pca_map, dtype=bool)

    probabilities = hist / total
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * bin_centers)
    mu_total = mu[-1]
    denominator = omega * (1.0 - omega)
    sigma_b = np.zeros_like(denominator)
    valid = denominator > 1e-12
    sigma_b[valid] = (mu_total * omega[valid] - mu[valid]) ** 2 / denominator[valid]
    threshold = float(bin_centers[int(np.argmax(sigma_b))])
    return pca_map > threshold


def save_overlay(
    base_image: np.ndarray,
    pca_map: np.ndarray,
    output_path: Path,
    alpha: float = 0.45,
) -> None:
    values = np.clip(pca_map, 0.0, 1.0)
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


def load_token_payload(tokens_root: Path, model_name: str, block_name: str) -> Dict[str, Any]:
    path = tokens_root / model_name / "tokens" / f"{block_name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing token file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tokens = payload["tokens"]
    if list(tokens.shape[:2]) != [len(payload["image_ids"]), 196]:
        raise ValueError(
            f"Unexpected token shape for {path}: {tuple(tokens.shape)} "
            f"with {len(payload['image_ids'])} image ids."
        )
    return payload


def process_block(
    tokens_root: Path,
    output_root: Path,
    manifest_paths: Dict[str, Path],
    model_name: str,
    block_name: str,
    reference_image_ids: List[str] | None,
    max_images: int | None = None,
) -> tuple[Dict[str, Any], List[str]]:
    payload = load_token_payload(tokens_root, model_name, block_name)
    image_ids = list(payload["image_ids"])
    tokens = payload["tokens"]
    if max_images is not None:
        image_ids = image_ids[:max_images]
        tokens = tokens[:max_images]

    if reference_image_ids is not None and image_ids != reference_image_ids:
        raise ValueError(f"Image ids mismatch for {model_name}/{block_name}.")

    out_dir = output_root / model_name / block_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pca_mins: List[float] = []
    pca_maxs: List[float] = []
    pca_means: List[float] = []
    nan_count = 0
    constant_count = 0
    map_count = 0
    top20_count = 0
    otsu_count = 0
    example_shape = None

    overlay_cache: Dict[str, np.ndarray] = {}
    for index, image_id in enumerate(tqdm(image_ids, desc=f"{model_name} {block_name} PCA")):
        if image_id not in manifest_paths:
            raise KeyError(f"{image_id} not found in manifest.")
        pca_map, constant = compute_pca_map(tokens[index])
        if np.isnan(pca_map).any():
            nan_count += 1
        if constant:
            constant_count += 1

        mask_top20 = top20_mask(pca_map, constant)
        mask_otsu = otsu_mask(pca_map, constant)

        np.save(out_dir / f"{image_id}.npy", pca_map)
        np.save(out_dir / f"{image_id}_mask_top20.npy", mask_top20)
        np.save(out_dir / f"{image_id}_mask_otsu.npy", mask_otsu)

        if image_id not in overlay_cache:
            overlay_cache[image_id] = load_overlay_image(manifest_paths[image_id])
        save_overlay(overlay_cache[image_id], pca_map, out_dir / f"{image_id}_overlay.png")

        pca_mins.append(float(np.min(pca_map)))
        pca_maxs.append(float(np.max(pca_map)))
        pca_means.append(float(np.mean(pca_map)))
        map_count += 1
        top20_count += 1
        otsu_count += 1
        example_shape = list(pca_map.shape)

    report = {
        "num_images_processed": map_count,
        "input_tokens_shape": list(tokens.shape),
        "pca_maps_saved": map_count,
        "example_pca_map_shape": example_shape,
        "pca_map_min": float(np.min(pca_mins)) if pca_mins else None,
        "pca_map_max": float(np.max(pca_maxs)) if pca_maxs else None,
        "pca_map_mean": float(np.mean(pca_means)) if pca_means else None,
        "num_maps_with_nan": nan_count,
        "num_constant_or_near_constant_maps": constant_count,
        "top20_masks_saved": top20_count,
        "otsu_masks_saved": otsu_count,
        "output_path": out_dir.as_posix(),
    }
    return report, image_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ViT PCA maps from saved patch tokens.")
    parser.add_argument("--tokens_root", default=DEFAULT_TOKENS_ROOT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens_root = Path(args.tokens_root)
    output_root = Path(args.output_dir)
    manifest_paths = read_manifest(Path(args.manifest))

    report: Dict[str, Dict[str, Any]] = {}
    image_id_references: Dict[str, List[str]] = {}
    global_reference: List[str] | None = None

    for model_name in MODELS:
        report[model_name] = {}
        model_reference: List[str] | None = None
        for block_name in BLOCKS:
            block_report, image_ids = process_block(
                tokens_root=tokens_root,
                output_root=output_root,
                manifest_paths=manifest_paths,
                model_name=model_name,
                block_name=block_name,
                reference_image_ids=model_reference,
                max_images=args.max_images,
            )
            if model_reference is None:
                model_reference = image_ids
            report[model_name][block_name] = block_report
        image_id_references[model_name] = model_reference or []
        if global_reference is None:
            global_reference = model_reference
        elif model_reference != global_reference:
            raise ValueError(f"Image ids mismatch between models at {model_name}.")

    report["image_ids_identical_between_models"] = (
        image_id_references["vit_supervised"] == image_id_references["vit_lejepa"]
    )
    report["tokens_root"] = tokens_root.as_posix()
    report["manifest"] = Path(args.manifest).as_posix()

    report_path = output_root / "vit_pca_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nPCA sanity report")
    for model_name in MODELS:
        print(f"\n{model_name}")
        for block_name in BLOCKS:
            block_report = report[model_name][block_name]
            print(
                f"  {block_name}: maps={block_report['pca_maps_saved']} "
                f"tokens={block_report['input_tokens_shape']} "
                f"map_shape={block_report['example_pca_map_shape']} "
                f"nan={block_report['num_maps_with_nan']} "
                f"constant={block_report['num_constant_or_near_constant_maps']} "
                f"mean={block_report['pca_map_mean']:.4f}"
            )
    print(f"\nimage_ids identical between models: {report['image_ids_identical_between_models']}")
    print(f"Report JSON: {report_path}")


if __name__ == "__main__":
    main()
