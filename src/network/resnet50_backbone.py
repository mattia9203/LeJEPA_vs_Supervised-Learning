from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNet50Backbone(nn.Module):
    feature_dim = 2048

    def __init__(self) -> None:
        super().__init__()
        model = resnet50(weights=None)
        self.stem = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                ]
            )
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward_intermediates(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return {
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
        }

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_intermediates(x)["layer4"]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.forward_feature_map(x)
        return torch.flatten(self.avgpool(feature_map), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
