import torch
import torch.nn as nn

from ..globals import NUM_CLASSES
from .classifier import ClassifierHead
from .resnet50_backbone import ResNet50Backbone


class SupervisedCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = False,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = ResNet50Backbone()
        self.head = ClassifierHead(
            ResNet50Backbone.feature_dim,
            num_classes,
            dropout=head_dropout,
        )
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_feature_map(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
