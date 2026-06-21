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
        trainable_last_blocks: int | None = None,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.trainable_last_blocks = trainable_last_blocks
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        embed_dim = self.backbone.num_features
        self.head = ClassifierHead(embed_dim, num_classes, dropout=head_dropout)

        if freeze_backbone:
            self._freeze_backbone()
        elif trainable_last_blocks is not None:
            self._unfreeze_last_blocks(trainable_last_blocks)

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters for linear probing."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def _unfreeze_last_blocks(self, num_blocks: int) -> None:
        """Train only the final transformer blocks and output normalization."""
        blocks = self.backbone.blocks
        if not 1 <= num_blocks <= len(blocks):
            raise ValueError(
                f"trainable_last_blocks must be between 1 and {len(blocks)}, "
                f"received {num_blocks}."
            )

        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        """Set training mode while keeping frozen backbone sections in eval."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        elif self.trainable_last_blocks is not None:
            self.backbone.eval()
            for block in self.backbone.blocks[-self.trainable_last_blocks:]:
                block.train(mode)
            self.backbone.norm.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: backbone features -> classifier logits."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits
