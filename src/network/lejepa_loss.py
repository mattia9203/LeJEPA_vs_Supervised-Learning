"""Official-style LeJEPA invariance and SIGReg objective."""

from typing import Dict

import torch
import torch.nn as nn

import lejepa


class LeJEPALoss(nn.Module):
    """LeJEPA loss without predictor, teacher, or stop-gradient.

    Projections have shape [views, batch, projection_dim]. The invariance term
    pulls every view toward the per-image mean projection. SIGReg is the
    official sliced Epps-Pulley normality statistic.
    """

    def __init__(
        self,
        sigreg_weight: float = 0.02,
        num_slices: int = 1024,
        num_points: int = 17,
    ) -> None:
        super().__init__()
        if not 0.0 <= sigreg_weight <= 1.0:
            raise ValueError("sigreg_weight must be in [0, 1].")
        self.sigreg_weight = sigreg_weight
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(n_points=num_points),
            num_slices=num_slices,
        )

    def forward(self, projections: torch.Tensor) -> Dict[str, torch.Tensor]:
        if projections.ndim != 3:
            raise ValueError(
                "LeJEPA projections must have shape [views, batch, dim], "
                f"received {tuple(projections.shape)}."
            )
        view_mean = projections.mean(dim=0, keepdim=True)
        invariance = (projections - view_mean).square().mean()
        sigreg = self.sigreg(projections.float())
        total = (1.0 - self.sigreg_weight) * invariance
        total = total + self.sigreg_weight * sigreg
        return {
            "total_loss": total,
            "invariance_loss": invariance,
            "sigreg_loss": sigreg,
        }
