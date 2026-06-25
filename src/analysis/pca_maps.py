"""PCA visualisation of patch-token representations.

Planned functionality:
    - Given patch-token features [B, 196, D] from a transformer layer,
      compute PCA (3 components) and reshape to [B, 14, 14, 3] colour maps.
    - Visualise the first 3 principal components as RGB.
    - Compare PCA maps across layers and between supervised / LeJEPA models.

References:
    - DINO paper (Caron et al., 2021) uses PCA of patch tokens to show
      that self-supervised ViTs learn semantically meaningful features.
"""

# TODO: Implement after feature extraction is working.


def compute_pca_maps(patch_features, n_components=3):
    """Compute PCA colour maps from patch-token features.

    Args:
        patch_features: Tensor of shape [B, 196, D]
        n_components: Number of PCA components (default 3 for RGB)

    Returns:
        PCA maps of shape [B, 14, 14, n_components]
    """
    raise NotImplementedError("PCA maps will be implemented in a later phase.")


def visualise_pca_map(pca_map, original_image=None, save_path=None):
    """Plot a single PCA map, optionally overlaid on the original image."""
    raise NotImplementedError("PCA visualisation will be implemented in a later phase.")
