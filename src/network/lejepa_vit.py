"""LeJEPA ViT-S/16 wrapper using Hugging Face transformers.

Checkpoint: OK-AI/lejepa-vits16-pretrain-in1k

LeJEPA is a self-supervised model. It does not have a classification head.
We treat it as a feature extractor and attach a fresh ClassifierHead.

Output handling:
    The model may return different output formats depending on the
    transformers version and checkpoint configuration. This wrapper
    inspects the output and extracts the [CLS] token embedding (or
    the first token of the last hidden state) for classification.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from ..globals import NUM_CLASSES, LEJEPA_VIT_MODEL
from .classifier import ClassifierHead


class LeJEPAViT(nn.Module):
    """LeJEPA ViT-S/16 from Hugging Face, with a linear classification head.

    Architecture:
        backbone  – HF AutoModel (pretrained, self-supervised)
        head      – ClassifierHead mapping embed_dim → num_classes
    """

    def __init__(
        self,
        model_name: str = LEJEPA_VIT_MODEL,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        # Load pretrained LeJEPA backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Determine embedding dimension
        embed_dim = self._get_embed_dim()

        # Fresh classification head
        self.head = ClassifierHead(embed_dim, num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    def _get_embed_dim(self) -> int:
        """Infer the embedding dimension from the model config."""
        # Try common config attribute names
        for attr in ("hidden_size", "embed_dim", "d_model"):
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        # Fallback: ViT-S default
        return 384

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Extracts the [CLS] token representation from the backbone
        and feeds it through the classification head.

        The output handling is robust: it tries several known output
        formats from Hugging Face models.
        """
        outputs = self.backbone(pixel_values=x)

        # --- Extract CLS / pooled features ---
        # Option 1: model returns pooler_output (common in HF ViTs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output  # [B, embed_dim]
        # Option 2: use last_hidden_state[:, 0] (CLS token)
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            features = outputs.last_hidden_state[:, 0]  # [B, embed_dim]
        # Option 3: plain tensor output
        elif isinstance(outputs, torch.Tensor):
            if outputs.dim() == 3:
                features = outputs[:, 0]  # assume [B, N, D], take CLS
            else:
                features = outputs  # assume [B, D]
        else:
            raise ValueError(
                f"Unexpected output type from LeJEPA backbone: {type(outputs)}. "
                f"Available keys/attributes: {dir(outputs)}"
            )

        logits = self.head(features)  # [B, num_classes]
        return logits
