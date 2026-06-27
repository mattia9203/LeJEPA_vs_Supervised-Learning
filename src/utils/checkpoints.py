import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    save_dir: str,
    is_best: bool = False,
    config: Optional[Dict[str, Any]] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }
    if hasattr(model, "backbone"):
        state["backbone_state_dict"] = model.backbone.state_dict()
    torch.save(state, os.path.join(save_dir, "last.pt"))
    if is_best:
        torch.save(state, os.path.join(save_dir, "best.pt"))


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if (
        scheduler is not None
        and ckpt.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt
