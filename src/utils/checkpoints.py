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
) -> None:
    """Save model checkpoint. Always saves 'last.pt'; if *is_best*, also saves 'best.pt'."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    torch.save(state, os.path.join(save_dir, "last.pt"))
    if is_best:
        torch.save(state, os.path.join(save_dir, "best.pt"))


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint. Returns the saved metadata dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
