"""ResNet-50 encoder and projector for LeJEPA pretraining."""

from typing import Dict, Sequence

import torch
import torch.nn as nn

from .resnet50_backbone import ResNet50Backbone


class ProjectionMLP(nn.Module):
    """Projection head used by the official LeJEPA formulation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeJEPACNN(nn.Module):
    """ResNet-50 backbone with a single symmetric LeJEPA projector."""

    def __init__(
        self,
        proj_hidden_dim: int = 2048,
        proj_output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.projector = ProjectionMLP(
            ResNet50Backbone.feature_dim,
            proj_hidden_dim,
            proj_output_dim,
        )

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_feature_map(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def _encode_view_group(
        self,
        views: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode same-resolution views in one backbone call."""
        if not views:
            raise ValueError("A LeJEPA view group cannot be empty.")
        batch_size = views[0].shape[0]
        combined = torch.cat(list(views), dim=0)
        embeddings = self.forward_features(combined)
        projections = self.projector(embeddings)
        num_views = len(views)
        return (
            embeddings.reshape(num_views, batch_size, -1),
            projections.reshape(num_views, batch_size, -1),
        )

    def forward_multicrop(
        self,
        global_views: Sequence[torch.Tensor],
        local_views: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Encode 2 global and 6 local views, preserving the view dimension."""
        global_embeddings, global_projections = self._encode_view_group(global_views)
        local_embeddings, local_projections = self._encode_view_group(local_views)
        return {
            "embeddings": torch.cat([global_embeddings, local_embeddings], dim=0),
            "projections": torch.cat([global_projections, local_projections], dim=0),
            "global_embeddings": global_embeddings,
            "local_embeddings": local_embeddings,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
