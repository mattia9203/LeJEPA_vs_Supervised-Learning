"""Semantic Alignment Score (SAS): correlation between PCA maps and XAI maps.

Planned functionality:
    - Given a PCA map and an XAI map for the same image, compute how well
      the structural features (PCA) align with the attention-based
      explanation (XAI).
    - Metrics: Pearson/Spearman correlation, or IoU after thresholding.
    - Compare SAS across layers, models, and training settings
      (linear probe vs. fine-tuned).

Interpretation:
    - High SAS → the model's attention aligns with learned feature structure.
    - Low SAS  → the model attends to regions that are structurally different
      from the dominant feature patterns.
"""

# TODO: Implement after PCA and XAI maps are working.


def compute_sas(pca_map, xai_map, method="pearson"):
    """Compute Semantic Alignment Score between a PCA map and an XAI map.

    Args:
        pca_map:  Tensor or array of shape [H, W] (e.g. first PCA component)
        xai_map:  Tensor or array of shape [H, W]
        method:   'pearson', 'spearman', or 'iou'

    Returns:
        float: Alignment score in [-1, 1] for correlation, [0, 1] for IoU.
    """
    raise NotImplementedError("SAS will be implemented in a later phase.")


def compute_sas_per_layer(pca_maps_by_layer, xai_maps_by_layer, method="pearson"):
    """Compute SAS for each transformer layer.

    Args:
        pca_maps_by_layer: dict {layer_idx: [H, W] maps}
        xai_maps_by_layer: dict {layer_idx: [H, W] maps}

    Returns:
        dict {layer_idx: mean_sas_score}
    """
    raise NotImplementedError("Per-layer SAS will be implemented in a later phase.")
