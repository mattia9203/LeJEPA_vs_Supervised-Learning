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


DEFAULT_FEATURES_ROOT = "outputs/xai/cnn/gradcam/features"
DEFAULT_MANIFEST = "outputs/manifests/analysis_val500_manifest.csv"
DEFAULT_OUTPUT_DIR = "outputs/pca/cnn"
DEFAULT_REPORTS_DIR = "outputs/pca/cnn/reports"
MODELS = ["cnn_supervised", "cnn_lejepa"]
LAYERS = ["layer2", "layer3", "layer4"]


def read_manifest(manifest_path: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    with open(manifest_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            mapping[f"image_{index:06d}"] = Path(row["path"])
    if not mapping:
        raise ValueError(f"No rows found in manifest: {manifest_path}")
    return mapping


def load_overlay_image(path: Path) -> np.ndarray:
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    image = Image.open(path).convert("RGB")
    return np.asarray(transform(image), dtype=np.uint8)


def normalize_map(values: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, bool]:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = values.min()
    max_value = values.max()
    if (max_value - min_value) <= eps:
        return torch.zeros_like(values), True
    return (values - min_value) / (max_value - min_value), False


def compute_pca_map(activation: torch.Tensor, eps: float = 1e-8) -> tuple[np.ndarray, bool]:
    if activation.dim() != 3:
        raise ValueError(f"Expected activation [C, H, W], got {tuple(activation.shape)}")
    channels, height, width = activation.shape
    tokens = activation.float().reshape(channels, height * width).transpose(0, 1)
    tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    if centered.norm() <= eps:
        return np.zeros((224, 224), dtype=np.float32), True

    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        pc1 = centered @ vh[0]
    except RuntimeError:
        torch.manual_seed(42)
        _, _, vh = torch.pca_lowrank(centered, q=1, center=False, niter=4)
        pc1 = centered @ vh[:, 0]

    spatial = pc1.abs().reshape(1, 1, height, width)
    upsampled = F.interpolate(
        spatial,
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
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * centers)
    mu_total = mu[-1]
    denominator = omega * (1.0 - omega)
    between = np.zeros_like(denominator)
    valid = denominator > 1e-12
    between[valid] = (mu_total * omega[valid] - mu[valid]) ** 2 / denominator[valid]
    threshold = float(centers[int(np.argmax(between))])
    return pca_map > threshold


def save_overlay(base_image: np.ndarray, pca_map: np.ndarray, output_path: Path) -> None:
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
    overlay = (0.55 * base) + (0.45 * heatmap)
    Image.fromarray(np.clip(overlay * 255.0, 0, 255).astype(np.uint8)).save(output_path)


def load_activation_payload(features_root: Path, model_name: str, layer_name: str) -> Dict[str, Any]:
    path = features_root / model_name / "activations" / f"{layer_name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing activation file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    activations = payload["activations"]
    if activations.dim() != 4:
        raise ValueError(f"Expected activations [N, C, H, W], got {tuple(activations.shape)}")
    if activations.shape[0] != len(payload["image_ids"]):
        raise ValueError(f"Activation/image_id count mismatch in {path}")
    return payload


def process_layer(
    features_root: Path,
    output_root: Path,
    manifest_paths: Dict[str, Path],
    model_name: str,
    layer_name: str,
    reference_image_ids: List[str] | None,
    max_images: int | None = None,
) -> tuple[Dict[str, Any], List[str]]:
    payload = load_activation_payload(features_root, model_name, layer_name)
    image_ids = list(payload["image_ids"])
    activations = payload["activations"]
    if max_images is not None:
        image_ids = image_ids[:max_images]
        activations = activations[:max_images]
    if reference_image_ids is not None and image_ids != reference_image_ids:
        raise ValueError(f"Image ids mismatch for {model_name}/{layer_name}.")

    out_dir = output_root / model_name / layer_name
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_cache: Dict[str, np.ndarray] = {}

    pca_mins: List[float] = []
    pca_maxs: List[float] = []
    pca_means: List[float] = []
    nan_count = 0
    constant_count = 0
    map_count = 0
    top20_count = 0
    otsu_count = 0
    example_shape = None

    for index, image_id in enumerate(tqdm(image_ids, desc=f"{model_name} {layer_name} PCA")):
        if image_id not in manifest_paths:
            raise KeyError(f"{image_id} not found in manifest.")
        pca_map, constant = compute_pca_map(activations[index])
        nan_count += int(np.isnan(pca_map).any())
        constant_count += int(constant)

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
        "input_activations_shape": list(activations.shape),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_root", default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reports_dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_root = Path(args.features_root)
    output_root = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    manifest_paths = read_manifest(Path(args.manifest))

    report: Dict[str, Any] = {}
    model_references: Dict[str, List[str]] = {}
    global_reference: List[str] | None = None

    for model_name in MODELS:
        report[model_name] = {}
        model_reference: List[str] | None = None
        for layer_name in LAYERS:
            layer_report, image_ids = process_layer(
                features_root,
                output_root,
                manifest_paths,
                model_name,
                layer_name,
                model_reference,
                max_images=args.max_images,
            )
            if model_reference is None:
                model_reference = image_ids
            report[model_name][layer_name] = layer_report
        model_references[model_name] = model_reference or []
        if global_reference is None:
            global_reference = model_reference
        elif model_reference != global_reference:
            raise ValueError(f"Image ids mismatch between models at {model_name}.")

    report["image_ids_identical_between_models"] = (
        model_references["cnn_supervised"] == model_references["cnn_lejepa"]
    )
    report["features_root"] = features_root.as_posix()
    report["manifest"] = Path(args.manifest).as_posix()

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "cnn_pca_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nCNN PCA sanity report")
    for model_name in MODELS:
        print(f"\n{model_name}")
        for layer_name in LAYERS:
            layer_report = report[model_name][layer_name]
            print(
                f"  {layer_name}: maps={layer_report['pca_maps_saved']} "
                f"activations={layer_report['input_activations_shape']} "
                f"map_shape={layer_report['example_pca_map_shape']} "
                f"nan={layer_report['num_maps_with_nan']} "
                f"constant={layer_report['num_constant_or_near_constant_maps']} "
                f"mean={layer_report['pca_map_mean']:.4f}"
            )
    print(f"\nimage_ids identical between models: {report['image_ids_identical_between_models']}")
    print(f"Report JSON: {report_path}")


if __name__ == "__main__":
    main()
