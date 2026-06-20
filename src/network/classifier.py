"""Classification head for ViT backbones."""

import torch.nn as nn


class ClassifierHead(nn.Module):
    """Simple linear classification head.

    Takes the [CLS] token embedding (or pooled representation) from a ViT
    backbone and maps it to *num_classes* logits.
    """

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)
