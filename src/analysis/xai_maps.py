"""Explainability (XAI) maps via Attention Rollout.

Planned functionality:
    - Implement Attention Rollout (Abnar & Zuidema, 2020) to aggregate
      attention across all transformer layers into a single attention map.
    - For each image, produce a [14, 14] spatial attention map showing
      which patches the model attends to.
    - Support both supervised ViT and LeJEPA ViT backbones.

References:
    - Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers"
"""

# TODO: Implement after feature extraction is working.


def attention_rollout(attentions, head_fusion="mean"):
    """Compute attention rollout from a list of attention matrices.

    Args:
        attentions: List of attention tensors, each [B, num_heads, N, N]
        head_fusion: How to fuse heads ('mean', 'max', 'min')

    Returns:
        Rollout attention map of shape [B, N] (attention to CLS token)
    """
    raise NotImplementedError("Attention rollout will be implemented in a later phase.")


def visualise_xai_map(xai_map, original_image=None, save_path=None):
    """Plot a spatial XAI attention map, optionally overlaid on the original."""
    raise NotImplementedError("XAI visualisation will be implemented in a later phase.")
