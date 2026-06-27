class FeatureExtractor:
    def __init__(self, model, layer_indices=None):
        self.model = model
        self.layer_indices = layer_indices or [3, 6, 9, 12]
        self._features = {}

    def extract(self, dataloader, device="cuda"):
        raise NotImplementedError(
            "Use vit_gmar_pipeline.py or cnn_step3_pipeline.py for the current analysis outputs."
        )
