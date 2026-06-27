from typing import Optional

import torch
import numpy as np


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return 100.0 * correct / targets.size(0)


def compute_f1(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    average: str = "macro",
) -> Optional[float]:
    try:
        from sklearn.metrics import f1_score
        return f1_score(all_targets, all_preds, average=average, zero_division=0)
    except ImportError:
        return None
