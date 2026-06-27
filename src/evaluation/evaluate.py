import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from ..utils.seed import set_seed
from ..utils.logging import setup_logger
from ..utils.metrics import compute_accuracy, compute_f1
from ..data.imagenet100 import get_imagenet100_loaders
from ..network.model_factory import create_model


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            preds = logits.argmax(dim=1)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    all_preds_np = np.concatenate(all_preds)
    all_targets_np = np.concatenate(all_targets)
    f1 = compute_f1(all_preds_np, all_targets_np)

    results = {
        "val_loss": avg_loss,
        "val_acc": avg_acc,
        "val_samples": total_samples,
    }
    if f1 is not None:
        results["val_f1_macro"] = f1
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.device:
        config["device"] = args.device

    set_seed(config.get("seed", 42))
    logger = setup_logger("eval")

    _, val_loader = get_imagenet100_loaders(config)
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    model = create_model(config)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    results = evaluate(model, val_loader, device, use_amp=config.get("mixed_precision", False))

    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    logger.info("=" * 50)

    exp_dir = os.path.join(config["output_dir"], config["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)
    results_path = os.path.join(exp_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
