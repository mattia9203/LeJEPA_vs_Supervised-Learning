"""Deterministic linear probing for frozen image backbones."""

import copy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils.metrics import compute_f1
from ..utils.seed import set_seed


class LinearProbeEvaluator:
    """Extract frozen features and train a fresh linear classifier."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        batch_size: int = 256,
        seed: int = 42,
        near_constant_std_threshold: float = 1e-4,
        compute_effective_rank: bool = True,
        effective_rank_max_samples: int = 2048,
    ) -> None:
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed
        self.near_constant_std_threshold = near_constant_std_threshold
        self.compute_effective_rank = compute_effective_rank
        self.effective_rank_max_samples = effective_rank_max_samples

    def _feature_diagnostics(self, features: torch.Tensor) -> Dict[str, float]:
        """Measure feature spread and dimensionality on deterministic inputs."""
        features = features.float()
        feature_std = features.std(dim=0, unbiased=False)
        diagnostics = {
            "feature_std_mean": feature_std.mean().item(),
            "feature_std_min": feature_std.min().item(),
            "feature_norm_mean": features.norm(dim=1).mean().item(),
            "num_near_constant_dims": int(
                (feature_std <= self.near_constant_std_threshold).sum().item()
            ),
        }

        if not self.compute_effective_rank:
            diagnostics["effective_rank"] = float("nan")
            return diagnostics

        rank_features = features
        max_samples = self.effective_rank_max_samples
        if max_samples and rank_features.shape[0] > max_samples:
            indices = torch.linspace(
                0,
                rank_features.shape[0] - 1,
                steps=max_samples,
            ).round().long()
            rank_features = rank_features[indices]

        centered = rank_features - rank_features.mean(dim=0, keepdim=True)
        gram = centered @ centered.T
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min_(0)
        eigenvalue_sum = eigenvalues.sum()
        if eigenvalue_sum <= torch.finfo(eigenvalues.dtype).eps:
            diagnostics["effective_rank"] = 0.0
        else:
            probabilities = eigenvalues / eigenvalue_sum
            probabilities = probabilities[probabilities > 0]
            entropy = -(probabilities * probabilities.log()).sum()
            diagnostics["effective_rank"] = entropy.exp().item()
        return diagnostics

    @staticmethod
    def _batchnorm_state(module: nn.Module) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        state = {}
        for name, child in module.named_modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                state[name] = (
                    child.running_mean.detach().clone(),
                    child.running_var.detach().clone(),
                )
        return state

    @torch.inference_mode()
    def _extract(
        self,
        backbone: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = []
        labels = []
        backbone.eval()
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            features.append(backbone.forward_features(images).float().cpu())
            labels.append(targets.cpu())
        return torch.cat(features), torch.cat(labels)

    def run(
        self,
        backbone: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        set_seed(self.seed)
        original_training = backbone.training
        original_requires_grad = [parameter.requires_grad for parameter in backbone.parameters()]
        before_bn = self._batchnorm_state(backbone)
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        backbone.eval()

        train_features, train_labels = self._extract(backbone, train_loader, device)
        val_features, val_labels = self._extract(backbone, val_loader, device)
        feature_diagnostics = self._feature_diagnostics(val_features)
        after_bn = self._batchnorm_state(backbone)
        for name, (before_mean, before_var) in before_bn.items():
            after_mean, after_var = after_bn[name]
            if not torch.equal(before_mean, after_mean) or not torch.equal(
                before_var,
                after_var,
            ):
                raise RuntimeError(f"Linear probe modified BatchNorm state at {name}.")

        probe = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.num_classes),
        ).to(device)
        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.epochs),
        )
        criterion = nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(self.seed)
        feature_loader = DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        for _ in range(self.epochs):
            probe.train()
            for features, targets in feature_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(probe(features), targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

        probe.eval()
        val_features = val_features.to(device)
        val_labels_device = val_labels.to(device)
        with torch.inference_mode():
            logits = probe(val_features)
            loss = criterion(logits, val_labels_device).item()
            predictions = logits.argmax(dim=1)
        accuracy = 100.0 * (predictions == val_labels_device).float().mean().item()
        f1 = compute_f1(predictions.cpu().numpy(), val_labels.numpy())

        for parameter, requires_grad in zip(backbone.parameters(), original_requires_grad):
            parameter.requires_grad = requires_grad
        backbone.train(original_training)
        metrics = {
            "probe_loss": loss,
            "probe_acc": accuracy,
            "probe_f1": float(f1) if f1 is not None else float("nan"),
            **feature_diagnostics,
        }
        return metrics, copy.deepcopy(probe.state_dict())
