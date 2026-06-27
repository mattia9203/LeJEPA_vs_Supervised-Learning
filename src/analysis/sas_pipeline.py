from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scipy.stats import mannwhitneyu, rankdata, wilcoxon
except ImportError:
    mannwhitneyu = None
    rankdata = None
    wilcoxon = None


DEFAULT_OUTPUT_DIR = "outputs/sas"
VIT_MODELS = ["vit_supervised", "vit_lejepa"]
CNN_MODELS = ["cnn_supervised", "cnn_lejepa"]
VIT_BLOCKS = ["block_03", "block_06", "block_09", "block_11"]
CNN_LAYERS = ["layer2", "layer3", "layer4"]
RAW_METRICS = ["pearson", "spearman", "iou_top20", "iou_otsu"]
SCORE_METRICS = RAW_METRICS + ["z_spearman", "z_iou_top20", "n_sas"]
EPS = 1e-8


@dataclass(frozen=True)
class GroupSpec:
    architecture: str
    model_name: str
    layer_or_block: str
    xai_method: str
    pca_dir: Path
    xai_dir: Path
    predictions_path: Path


@dataclass
class LoadedGroup:
    spec: GroupSpec
    image_ids: List[str]
    metadata: pd.DataFrame
    pca_maps: np.ndarray
    xai_maps: np.ndarray
    report: Dict[str, Any]


def as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def resolve_existing_path(candidates: Sequence[str | Path], description: str) -> Path:
    checked = [as_path(path) for path in candidates]
    for path in checked:
        if path.exists():
            return path
    rendered = "\n  ".join(path.as_posix() for path in checked)
    raise FileNotFoundError(f"Could not find {description}. Checked:\n  {rendered}")


def resolve_predictions_path(architecture: str) -> Path:
    if architecture == "vit":
        return resolve_existing_path(
            [
                "outputs/xai/vit/gmar/metadata/vit_predictions.csv",
                "outputs/analysis/gmar/vit/metadata/vit_predictions.csv",
                "outputs/analysis/metadata/vit_predictions.csv",
                "analysis_outputs/metadata/vit_predictions.csv",
                "analysis_outputs/vit_predictions.csv",
            ],
            "ViT predictions CSV",
        )
    if architecture == "cnn":
        return resolve_existing_path(
            [
                "outputs/xai/cnn/gradcam/metadata/cnn_predictions.csv",
                "outputs/analysis/cnn/metadata/cnn_predictions.csv",
                "outputs/analysis/metadata/cnn_predictions.csv",
                "analysis_outputs/metadata/cnn_predictions.csv",
                "analysis_outputs/cnn_predictions.csv",
            ],
            "CNN predictions CSV",
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def resolve_pca_dir(architecture: str, model_name: str, layer_or_block: str) -> Path:
    if architecture == "vit":
        candidates = [
            Path("outputs/pca/vit") / model_name / layer_or_block,
            Path("outputs/analysis/pca/vit") / model_name / layer_or_block,
            Path("outputs/analysis/pca") / model_name / layer_or_block,
            Path("analysis_outputs/pca/vit") / model_name / layer_or_block,
            Path("analysis_outputs/pca") / model_name / layer_or_block,
        ]
    else:
        candidates = [
            Path("outputs/pca/cnn") / model_name / layer_or_block,
            Path("outputs/analysis/cnn/pca") / model_name / layer_or_block,
            Path("outputs/analysis/pca") / model_name / layer_or_block,
            Path("analysis_outputs/cnn/pca") / model_name / layer_or_block,
            Path("analysis_outputs/pca") / model_name / layer_or_block,
        ]
    return resolve_existing_path(candidates, f"PCA maps for {model_name}/{layer_or_block}")


def resolve_xai_dir(architecture: str, model_name: str, layer_or_block: str) -> Path:
    if architecture == "vit":
        candidates = [
            Path("outputs/xai/vit/gmar/saliency") / model_name / "gmar" / layer_or_block,
            Path("outputs/analysis/gmar/vit/saliency") / model_name / "gmar" / layer_or_block,
            Path("outputs/analysis/saliency") / model_name / "gmar" / layer_or_block,
            Path("analysis_outputs/gmar/vit/saliency") / model_name / "gmar" / layer_or_block,
            Path("analysis_outputs/saliency") / model_name / "gmar" / layer_or_block,
        ]
    else:
        candidates = [
            Path("outputs/xai/cnn/gradcam/saliency") / model_name / "gradcam" / layer_or_block,
            Path("outputs/analysis/cnn/saliency") / model_name / "gradcam" / layer_or_block,
            Path("outputs/analysis/saliency") / model_name / "gradcam" / layer_or_block,
            Path("analysis_outputs/cnn/saliency") / model_name / "gradcam" / layer_or_block,
            Path("analysis_outputs/saliency") / model_name / "gradcam" / layer_or_block,
        ]
    return resolve_existing_path(candidates, f"XAI maps for {model_name}/{layer_or_block}")


def build_group_specs() -> List[GroupSpec]:
    vit_predictions = resolve_predictions_path("vit")
    cnn_predictions = resolve_predictions_path("cnn")
    specs: List[GroupSpec] = []
    for model_name in VIT_MODELS:
        for block_name in VIT_BLOCKS:
            specs.append(
                GroupSpec(
                    architecture="vit",
                    model_name=model_name,
                    layer_or_block=block_name,
                    xai_method="gmar",
                    pca_dir=resolve_pca_dir("vit", model_name, block_name),
                    xai_dir=resolve_xai_dir("vit", model_name, block_name),
                    predictions_path=vit_predictions,
                )
            )
    for model_name in CNN_MODELS:
        for layer_name in CNN_LAYERS:
            specs.append(
                GroupSpec(
                    architecture="cnn",
                    model_name=model_name,
                    layer_or_block=layer_name,
                    xai_method="gradcam",
                    pca_dir=resolve_pca_dir("cnn", model_name, layer_name),
                    xai_dir=resolve_xai_dir("cnn", model_name, layer_name),
                    predictions_path=cnn_predictions,
                )
            )
    return specs


def stable_group_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], byteorder="little")) % (2**32)


def read_predictions(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "image_id",
        "image_path",
        "true_label",
        "model_name",
        "predicted_label",
        "confidence",
        "correct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = frame[frame["model_name"] == model_name].copy()
    if frame.empty:
        raise ValueError(f"No prediction rows for {model_name} in {path}")
    frame["correct"] = frame["correct"].astype(int)
    return frame


def map_path(directory: Path, image_id: str) -> Path:
    return directory / f"{image_id}.npy"


def load_map(path: Path) -> np.ndarray:
    values = np.load(path)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D map at {path}, got shape {values.shape}")
    return values.astype(np.float32, copy=False)


def load_group(spec: GroupSpec, max_images: int | None = None) -> LoadedGroup:
    predictions = read_predictions(spec.predictions_path, spec.model_name)
    if max_images is not None:
        predictions = predictions.head(max_images).copy()

    expected_ids = predictions["image_id"].astype(str).tolist()
    pca_found = 0
    xai_found = 0
    missing_pca: List[str] = []
    missing_xai: List[str] = []
    image_ids: List[str] = []
    pca_maps: List[np.ndarray] = []
    xai_maps: List[np.ndarray] = []
    pca_nan_maps = 0
    xai_nan_maps = 0
    pca_constant_maps = 0
    xai_constant_maps = 0
    example_shape: List[int] | None = None

    for image_id in tqdm(
        expected_ids,
        desc=f"Loading {spec.model_name} {spec.layer_or_block}",
        leave=False,
    ):
        pca_file = map_path(spec.pca_dir, image_id)
        xai_file = map_path(spec.xai_dir, image_id)
        pca_exists = pca_file.exists()
        xai_exists = xai_file.exists()
        pca_found += int(pca_exists)
        xai_found += int(xai_exists)
        if not pca_exists:
            missing_pca.append(image_id)
        if not xai_exists:
            missing_xai.append(image_id)
        if not (pca_exists and xai_exists):
            continue

        pca_map = load_map(pca_file)
        xai_map = load_map(xai_file)
        if pca_map.shape != xai_map.shape:
            raise ValueError(
                f"Shape mismatch for {spec.model_name}/{spec.layer_or_block}/{image_id}: "
                f"PCA {pca_map.shape}, XAI {xai_map.shape}"
            )
        if pca_map.shape != (224, 224) or xai_map.shape != (224, 224):
            raise ValueError(
                f"Expected [224,224] maps for {spec.model_name}/{spec.layer_or_block}/{image_id}, "
                f"got {pca_map.shape} and {xai_map.shape}"
            )
        example_shape = list(pca_map.shape)

        pca_nan_maps += int(np.isnan(pca_map).any())
        xai_nan_maps += int(np.isnan(xai_map).any())
        pca_constant_maps += int(is_constant_map(pca_map))
        xai_constant_maps += int(is_constant_map(xai_map))
        image_ids.append(image_id)
        pca_maps.append(pca_map)
        xai_maps.append(xai_map)

    if not image_ids:
        raise ValueError(f"No paired PCA/XAI maps found for {spec.model_name}/{spec.layer_or_block}")

    metadata = predictions[predictions["image_id"].astype(str).isin(image_ids)].copy()
    metadata["_order"] = pd.Categorical(metadata["image_id"].astype(str), categories=image_ids, ordered=True)
    metadata = metadata.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    if metadata["image_id"].astype(str).tolist() != image_ids:
        raise ValueError(f"Prediction/image_id order mismatch for {spec.model_name}/{spec.layer_or_block}")

    report = {
        "num_expected_from_predictions": len(expected_ids),
        "num_images_processed": len(image_ids),
        "num_pca_maps_found": pca_found,
        "num_xai_maps_found": xai_found,
        "num_missing_pca_maps": len(missing_pca),
        "num_missing_xai_maps": len(missing_xai),
        "missing_pca_examples": missing_pca[:10],
        "missing_xai_examples": missing_xai[:10],
        "num_nan_pca_maps": pca_nan_maps,
        "num_nan_xai_maps": xai_nan_maps,
        "num_constant_pca_maps": pca_constant_maps,
        "num_constant_xai_maps": xai_constant_maps,
        "example_map_shape": example_shape,
        "pca_dir": spec.pca_dir.as_posix(),
        "xai_dir": spec.xai_dir.as_posix(),
        "predictions_path": spec.predictions_path.as_posix(),
        "image_ids_coherent_with_predictions": image_ids == expected_ids[: len(image_ids)]
        if not missing_pca and not missing_xai
        else set(image_ids).issubset(set(expected_ids)),
    }

    return LoadedGroup(
        spec=spec,
        image_ids=image_ids,
        metadata=metadata,
        pca_maps=np.stack(pca_maps, axis=0),
        xai_maps=np.stack(xai_maps, axis=0),
        report=report,
    )


def is_constant_map(values: np.ndarray, eps: float = EPS) -> bool:
    if np.isnan(values).any():
        return False
    return bool((np.max(values) - np.min(values)) <= eps)


def downsample_mean(maps: np.ndarray, target_size: int) -> np.ndarray:
    if maps.shape[-2:] == (target_size, target_size):
        return maps
    height, width = maps.shape[-2:]
    if height % target_size != 0 or width % target_size != 0:
        raise ValueError(f"Cannot mean-pool maps with shape {(height, width)} to {target_size}.")
    scale_h = height // target_size
    scale_w = width // target_size
    return maps.reshape(maps.shape[0], target_size, scale_h, target_size, scale_w).mean(axis=(2, 4))


def numpy_rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def rank_rows(values: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(values, dtype=np.float32)
    for index in range(values.shape[0]):
        if rankdata is not None:
            ranks[index] = rankdata(values[index], method="average").astype(np.float32)
        else:
            ranks[index] = numpy_rankdata(values[index]).astype(np.float32)
    return ranks


def normalized_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values).all(axis=1)
    means = np.nanmean(values, axis=1, keepdims=True)
    centered = values - means
    norms = np.linalg.norm(centered, axis=1)
    valid = finite & (norms > EPS)
    normalized = np.zeros_like(centered, dtype=np.float32)
    normalized[valid] = (centered[valid] / norms[valid, None]).astype(np.float32)
    return normalized, valid


def pairwise_dot(
    left: np.ndarray,
    left_valid: np.ndarray,
    right: np.ndarray,
    right_valid: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
) -> np.ndarray:
    values = np.full(left_idx.shape[0], np.nan, dtype=np.float32)
    valid = left_valid[left_idx] & right_valid[right_idx]
    if np.any(valid):
        values[valid] = np.sum(left[left_idx[valid]] * right[right_idx[valid]], axis=1)
    return values


def top_fraction_masks(maps: np.ndarray, fraction: float = 0.20) -> np.ndarray:
    flat = np.nan_to_num(maps.reshape(maps.shape[0], -1), nan=-np.inf)
    count = max(1, int(math.ceil(flat.shape[1] * fraction)))
    selected = np.argpartition(flat, -count, axis=1)[:, -count:]
    masks = np.zeros(flat.shape, dtype=bool)
    rows = np.arange(flat.shape[0])[:, None]
    masks[rows, selected] = True
    return masks


def otsu_threshold(values: np.ndarray) -> float | None:
    if np.isnan(values).any() or is_constant_map(values):
        return None
    hist, bin_edges = np.histogram(values.reshape(-1), bins=256, range=(0.0, 1.0))
    total = hist.sum()
    if total <= 0:
        return None
    probabilities = hist.astype(np.float64) / float(total)
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * centers)
    denominator = omega * (1.0 - omega)
    between = np.zeros_like(denominator)
    valid = denominator > EPS
    between[valid] = (mu[-1] * omega[valid] - mu[valid]) ** 2 / denominator[valid]
    return float(centers[int(np.argmax(between))])


def otsu_masks(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_masks = np.zeros((maps.shape[0], maps.shape[1] * maps.shape[2]), dtype=bool)
    valid = np.zeros(maps.shape[0], dtype=bool)
    for index, values in enumerate(maps):
        threshold = otsu_threshold(values)
        if threshold is None:
            continue
        flat_masks[index] = (values.reshape(-1) > threshold)
        valid[index] = True
    return flat_masks, valid


def mask_iou(
    left: np.ndarray,
    left_valid: np.ndarray,
    right: np.ndarray,
    right_valid: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
) -> np.ndarray:
    values = np.full(left_idx.shape[0], np.nan, dtype=np.float32)
    valid = left_valid[left_idx] & right_valid[right_idx]
    if not np.any(valid):
        return values
    left_sel = left[left_idx[valid]]
    right_sel = right[right_idx[valid]]
    intersection = np.logical_and(left_sel, right_sel).sum(axis=1)
    union = np.logical_or(left_sel, right_sel).sum(axis=1)
    nonzero = union > 0
    valid_positions = np.where(valid)[0]
    values[valid_positions[nonzero]] = intersection[nonzero] / union[nonzero]
    return values


def shuffled_indices(num_images: int, shuffles_per_image: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    left_idx = np.repeat(np.arange(num_images), shuffles_per_image)
    if num_images < 2:
        raise ValueError("At least two images are required for shuffled baseline.")
    raw = rng.integers(0, num_images - 1, size=left_idx.shape[0])
    right_idx = raw + (raw >= left_idx)
    return left_idx.astype(np.int64), right_idx.astype(np.int64)


def metric_summary(values: np.ndarray) -> Dict[str, float | int | None]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"count": 0, "mean": None, "std": None, "median": None}
    return {
        "count": int(valid.size),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
        "median": float(np.median(valid)),
    }


def compute_group_metrics(
    group: LoadedGroup,
    num_shuffles_per_image: int,
    seed: int,
    corr_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    spec = group.spec
    num_images = len(group.image_ids)
    real_idx = np.arange(num_images, dtype=np.int64)
    pca_flat_corr = downsample_mean(group.pca_maps, corr_size).reshape(num_images, -1)
    xai_flat_corr = downsample_mean(group.xai_maps, corr_size).reshape(num_images, -1)

    pca_pearson, pca_pearson_valid = normalized_rows(pca_flat_corr)
    xai_pearson, xai_pearson_valid = normalized_rows(xai_flat_corr)
    pca_spearman, pca_spearman_valid = normalized_rows(rank_rows(pca_flat_corr))
    xai_spearman, xai_spearman_valid = normalized_rows(rank_rows(xai_flat_corr))

    pca_top20 = top_fraction_masks(group.pca_maps)
    xai_top20 = top_fraction_masks(group.xai_maps)
    top20_valid = np.ones(num_images, dtype=bool)

    pca_otsu, pca_otsu_valid = otsu_masks(group.pca_maps)
    xai_otsu, xai_otsu_valid = otsu_masks(group.xai_maps)

    real_values = {
        "pearson": pairwise_dot(
            pca_pearson,
            pca_pearson_valid,
            xai_pearson,
            xai_pearson_valid,
            real_idx,
            real_idx,
        ),
        "spearman": pairwise_dot(
            pca_spearman,
            pca_spearman_valid,
            xai_spearman,
            xai_spearman_valid,
            real_idx,
            real_idx,
        ),
        "iou_top20": mask_iou(
            pca_top20,
            top20_valid,
            xai_top20,
            top20_valid,
            real_idx,
            real_idx,
        ),
        "iou_otsu": mask_iou(
            pca_otsu,
            pca_otsu_valid,
            xai_otsu,
            xai_otsu_valid,
            real_idx,
            real_idx,
        ),
    }

    rng = np.random.default_rng(
        stable_group_seed(seed, spec.architecture, spec.model_name, spec.layer_or_block)
    )
    shuffle_left, shuffle_right = shuffled_indices(num_images, num_shuffles_per_image, rng)
    shuffled_values = {
        "pearson": pairwise_dot(
            pca_pearson,
            pca_pearson_valid,
            xai_pearson,
            xai_pearson_valid,
            shuffle_left,
            shuffle_right,
        ),
        "spearman": pairwise_dot(
            pca_spearman,
            pca_spearman_valid,
            xai_spearman,
            xai_spearman_valid,
            shuffle_left,
            shuffle_right,
        ),
        "iou_top20": mask_iou(
            pca_top20,
            top20_valid,
            xai_top20,
            top20_valid,
            shuffle_left,
            shuffle_right,
        ),
        "iou_otsu": mask_iou(
            pca_otsu,
            pca_otsu_valid,
            xai_otsu,
            xai_otsu_valid,
            shuffle_left,
            shuffle_right,
        ),
    }

    baseline_records: List[Dict[str, Any]] = []
    baseline_stats: Dict[str, Dict[str, Any]] = {}
    near_zero_std_metrics: List[str] = []
    for metric in RAW_METRICS:
        real_summary = metric_summary(real_values[metric])
        shuffled_summary = metric_summary(shuffled_values[metric])
        shuffled_std = shuffled_summary["std"]
        if shuffled_std is None or shuffled_std <= EPS:
            near_zero_std_metrics.append(metric)

        p_value = np.nan
        if mannwhitneyu is not None:
            real_valid = real_values[metric][np.isfinite(real_values[metric])]
            shuffled_valid = shuffled_values[metric][np.isfinite(shuffled_values[metric])]
            if real_valid.size > 0 and shuffled_valid.size > 0:
                try:
                    p_value = float(mannwhitneyu(real_valid, shuffled_valid, alternative="two-sided").pvalue)
                except ValueError:
                    p_value = np.nan

        baseline_stats[metric] = {
            "shuffled_mean": shuffled_summary["mean"],
            "shuffled_std": shuffled_summary["std"],
            "shuffled_median": shuffled_summary["median"],
            "shuffled_count": shuffled_summary["count"],
        }
        baseline_records.append(
            {
                "architecture": spec.architecture,
                "model_name": spec.model_name,
                "layer_or_block": spec.layer_or_block,
                "metric": metric,
                "real_mean": real_summary["mean"],
                "real_std": real_summary["std"],
                "shuffled_mean": shuffled_summary["mean"],
                "shuffled_std": shuffled_summary["std"],
                "shuffled_median": shuffled_summary["median"],
                "real_minus_shuffled": None
                if real_summary["mean"] is None or shuffled_summary["mean"] is None
                else float(real_summary["mean"] - shuffled_summary["mean"]),
                "shuffled_count": shuffled_summary["count"],
                "p_value_real_vs_shuffled": p_value,
            }
        )

    spearman_mean = baseline_stats["spearman"]["shuffled_mean"]
    spearman_std = baseline_stats["spearman"]["shuffled_std"]
    top20_mean = baseline_stats["iou_top20"]["shuffled_mean"]
    top20_std = baseline_stats["iou_top20"]["shuffled_std"]
    spearman_den = EPS if spearman_std is None or spearman_std <= EPS else float(spearman_std)
    top20_den = EPS if top20_std is None or top20_std <= EPS else float(top20_std)
    z_spearman = (real_values["spearman"] - float(spearman_mean)) / spearman_den
    z_iou_top20 = (real_values["iou_top20"] - float(top20_mean)) / top20_den
    n_sas = 0.5 * z_spearman + 0.5 * z_iou_top20
    n_sas[~(np.isfinite(z_spearman) & np.isfinite(z_iou_top20))] = np.nan

    score_frame = group.metadata.copy()
    score_frame.insert(3, "architecture", spec.architecture)
    score_frame.insert(5, "layer_or_block", spec.layer_or_block)
    score_frame.insert(6, "xai_method", spec.xai_method)
    score_frame["pearson"] = real_values["pearson"]
    score_frame["spearman"] = real_values["spearman"]
    score_frame["iou_top20"] = real_values["iou_top20"]
    score_frame["iou_otsu"] = real_values["iou_otsu"]
    score_frame["z_spearman"] = z_spearman
    score_frame["z_iou_top20"] = z_iou_top20
    score_frame["n_sas"] = n_sas
    score_frame = score_frame[
        [
            "image_id",
            "image_path",
            "true_label",
            "architecture",
            "model_name",
            "layer_or_block",
            "xai_method",
            "predicted_label",
            "confidence",
            "correct",
            "pearson",
            "spearman",
            "iou_top20",
            "iou_otsu",
            "z_spearman",
            "z_iou_top20",
            "n_sas",
        ]
    ]

    group_report = dict(group.report)
    group_report.update(
        {
            "num_metric_nan": {
                metric: int(np.isnan(score_frame[metric].to_numpy(dtype=float)).sum())
                for metric in SCORE_METRICS
            },
            "shuffled_baseline_count": {
                metric: int(baseline_stats[metric]["shuffled_count"]) for metric in RAW_METRICS
            },
            "shuffled_std_near_zero_metrics": near_zero_std_metrics,
            "mean_n_sas": float(np.nanmean(n_sas)) if np.isfinite(n_sas).any() else None,
            "mean_spearman": float(np.nanmean(real_values["spearman"]))
            if np.isfinite(real_values["spearman"]).any()
            else None,
            "mean_iou_top20": float(np.nanmean(real_values["iou_top20"]))
            if np.isfinite(real_values["iou_top20"]).any()
            else None,
        }
    )
    return score_frame, pd.DataFrame(baseline_records), group_report


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float | None, float | None]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return None, None
    if valid.size == 1 or samples <= 0:
        value = float(valid[0])
        return value, value
    draw_idx = rng.integers(0, valid.size, size=(samples, valid.size))
    means = valid[draw_idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_summary(scores: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for (architecture, model_name, layer_or_block), group in scores.groupby(
        ["architecture", "model_name", "layer_or_block"],
        sort=False,
    ):
        subsets = {
            "all": group,
            "correct_only": group[group["correct"].astype(int) == 1],
        }
        for subset_name, subset in subsets.items():
            for metric in SCORE_METRICS:
                values = subset[metric].to_numpy(dtype=float)
                summary = metric_summary(values)
                ci_low, ci_high = bootstrap_ci(values, rng, bootstrap_samples)
                rows.append(
                    {
                        "architecture": architecture,
                        "model_name": model_name,
                        "layer_or_block": layer_or_block,
                        "subset": subset_name,
                        "metric": metric,
                        "count": summary["count"],
                        "mean": summary["mean"],
                        "std": summary["std"],
                        "median": summary["median"],
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                    }
                )
    return pd.DataFrame(rows)


def wilcoxon_test(supervised: np.ndarray, lejepa: np.ndarray) -> tuple[float, float, float]:
    diff = lejepa - supervised
    finite = np.isfinite(diff)
    diff = diff[finite]
    if diff.size == 0:
        return np.nan, np.nan, np.nan
    effect = np.nan
    if diff.size > 1:
        diff_std = np.std(diff, ddof=1)
        if diff_std > EPS:
            effect = float(np.mean(diff) / diff_std)
    if np.all(np.abs(diff) <= EPS):
        return 0.0, 1.0, effect
    if wilcoxon is None:
        return np.nan, np.nan, effect
    try:
        result = wilcoxon(lejepa[finite], supervised[finite], zero_method="wilcox", alternative="two-sided")
        return float(result.statistic), float(result.pvalue), effect
    except ValueError:
        return np.nan, np.nan, effect


def build_paired_tests(scores: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    comparisons = [
        ("vit", "vit_supervised", "vit_lejepa", VIT_BLOCKS),
        ("cnn", "cnn_supervised", "cnn_lejepa", CNN_LAYERS),
    ]
    for architecture, supervised_model, lejepa_model, layers in comparisons:
        arch_scores = scores[scores["architecture"] == architecture]
        for layer_or_block in layers:
            layer_scores = arch_scores[arch_scores["layer_or_block"] == layer_or_block]
            sup = layer_scores[layer_scores["model_name"] == supervised_model].set_index("image_id")
            lej = layer_scores[layer_scores["model_name"] == lejepa_model].set_index("image_id")
            common_ids = sup.index.intersection(lej.index)
            sup = sup.loc[common_ids]
            lej = lej.loc[common_ids]
            subsets = {
                "all": np.ones(len(common_ids), dtype=bool),
                "both_correct": (sup["correct"].astype(int).to_numpy() == 1)
                & (lej["correct"].astype(int).to_numpy() == 1),
            }
            for subset_name, mask in subsets.items():
                for metric in ["n_sas", "spearman", "iou_top20", "pearson", "iou_otsu"]:
                    supervised_values = sup[metric].to_numpy(dtype=float)[mask]
                    lejepa_values = lej[metric].to_numpy(dtype=float)[mask]
                    finite = np.isfinite(supervised_values) & np.isfinite(lejepa_values)
                    supervised_values = supervised_values[finite]
                    lejepa_values = lejepa_values[finite]
                    statistic, p_value, effect = wilcoxon_test(supervised_values, lejepa_values)
                    diff = lejepa_values - supervised_values
                    rows.append(
                        {
                            "architecture": architecture,
                            "layer_or_block": layer_or_block,
                            "metric": metric,
                            "subset": subset_name,
                            "n_pairs": int(diff.size),
                            "supervised_mean": float(np.mean(supervised_values))
                            if supervised_values.size
                            else np.nan,
                            "lejepa_mean": float(np.mean(lejepa_values)) if lejepa_values.size else np.nan,
                            "mean_difference": float(np.mean(diff)) if diff.size else np.nan,
                            "median_difference": float(np.median(diff)) if diff.size else np.nan,
                            "wilcoxon_statistic": statistic,
                            "p_value": p_value,
                            "effect_size": effect,
                        }
                    )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    group_reports: Dict[str, Dict[str, Dict[str, Any]]],
    scores: pd.DataFrame,
    paths: Dict[str, str],
) -> Path:
    mean_n_sas: Dict[str, Dict[str, float | None]] = {}
    for (model_name, layer_or_block), group in scores.groupby(["model_name", "layer_or_block"], sort=False):
        mean_n_sas.setdefault(model_name, {})[layer_or_block] = (
            float(np.nanmean(group["n_sas"].to_numpy(dtype=float)))
            if np.isfinite(group["n_sas"].to_numpy(dtype=float)).any()
            else None
        )

    report = {
        "groups": group_reports,
        "mean_n_sas_by_model_layer": mean_n_sas,
        "csv_outputs": paths,
        "notes": {
            "n_sas": "0.5 * z_spearman + 0.5 * z_iou_top20, normalized against per-model/layer shuffled baselines.",
            "pipeline": "Read-only post-training analysis; PCA, GMAR, and Grad-CAM maps were not recalculated.",
        },
    }
    report_path = output_dir / "sas_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--num_shuffles_per_image", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corr_size", type=int, default=56)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_group_specs()
    all_scores: List[pd.DataFrame] = []
    all_baselines: List[pd.DataFrame] = []
    group_reports: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for spec in tqdm(specs, desc="SAS groups"):
        group = load_group(spec, max_images=args.max_images)
        scores, baseline, report = compute_group_metrics(
            group,
            num_shuffles_per_image=args.num_shuffles_per_image,
            seed=args.seed,
            corr_size=args.corr_size,
        )
        all_scores.append(scores)
        all_baselines.append(baseline)
        group_reports.setdefault(spec.model_name, {})[spec.layer_or_block] = report

    scores_frame = pd.concat(all_scores, ignore_index=True)
    baseline_frame = pd.concat(all_baselines, ignore_index=True)
    summary_frame = build_summary(scores_frame, args.bootstrap_samples, args.seed)
    paired_frame = build_paired_tests(scores_frame)

    scores_path = output_dir / "sas_scores.csv"
    summary_path = output_dir / "sas_summary_by_model_layer.csv"
    paired_path = output_dir / "sas_paired_tests.csv"
    baseline_path = output_dir / "sas_shuffled_baseline.csv"

    scores_frame.to_csv(scores_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    paired_frame.to_csv(paired_path, index=False)
    baseline_frame.to_csv(baseline_path, index=False)

    report_path = write_report(
        output_dir,
        group_reports,
        scores_frame,
        {
            "sas_scores": scores_path.as_posix(),
            "sas_summary_by_model_layer": summary_path.as_posix(),
            "sas_paired_tests": paired_path.as_posix(),
            "sas_shuffled_baseline": baseline_path.as_posix(),
        },
    )

    print("\nSAS sanity report")
    print(f"Rows in sas_scores.csv: {len(scores_frame)}")
    print(f"Rows in shuffled baseline CSV: {len(baseline_frame)}")
    print(f"Rows in paired tests CSV: {len(paired_frame)}")
    print("\nMean N-SAS by model/layer:")
    for (model_name, layer_or_block), group in scores_frame.groupby(["model_name", "layer_or_block"], sort=False):
        mean_value = float(np.nanmean(group["n_sas"].to_numpy(dtype=float)))
        print(f"  {model_name} {layer_or_block}: {mean_value:.4f}")
    print(f"\nReport JSON: {report_path}")


if __name__ == "__main__":
    main()
