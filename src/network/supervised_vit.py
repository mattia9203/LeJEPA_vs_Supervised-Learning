"""Supervised ViT-S/16 wrapper using timm."""

import torch
import torch.nn as nn
import timm

from ..globals import NUM_CLASSES, SUPERVISED_VIT_MODEL
from .classifier import ClassifierHead


class SupervisedViT(nn.Module):
    """Supervised pretrained ViT-S/16 from timm.

    The original timm classification head is removed and replaced with a
    fresh head for the selected ImageNet-100 subset classes.
    """

    def __init__(
        self,
        model_name: str = SUPERVISED_VIT_MODEL,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        embed_dim = self.backbone.num_features
        self.head = ClassifierHead(embed_dim, num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters for linear probing."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: backbone features -> classifier logits."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits
