"""Feature extraction from intermediate transformer layers.

Planned functionality:
    - Hook into specified transformer blocks (e.g., layers 3, 6, 9, 12)
    - Extract patch token activations [B, 196, D] for PCA/XAI analysis
    - Extract CLS token activations for representation comparison
    - Support both supervised ViT and LeJEPA ViT backbones

Usage (planned):
    extractor = FeatureExtractor(model, layer_indices=[3, 6, 9, 12])
    features = extractor.extract(dataloader)
"""

# TODO: Implement after core training pipeline is validated.


class FeatureExtractor:
    """Extract intermediate features from ViT transformer blocks.

    Registers forward hooks on specified layers to capture activations
    during a forward pass.
    """

    def __init__(self, model, layer_indices=None):
        self.model = model
        self.layer_indices = layer_indices or [3, 6, 9, 12]
        self._features = {}
        # TODO: Register hooks on the appropriate model layers.

    def extract(self, dataloader, device="cuda"):
        """Run the model on the dataloader and collect intermediate features.

        Returns:
            dict mapping layer_index → tensor of shape [N, num_tokens, embed_dim]
        """
        raise NotImplementedError("Feature extraction will be implemented after training pipeline validation.")
