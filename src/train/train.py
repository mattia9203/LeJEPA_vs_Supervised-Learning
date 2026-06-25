"""Training entry point.

Usage:
    python -m src.train.train --config configs/linear_probe_supervised.yaml
    python -m src.train.train --config configs/finetune_lejepa.yaml
"""

import argparse
import copy
import os
import torch
import yaml

from ..utils.seed import set_seed
from ..utils.logging import setup_logger
from ..data.imagenet100 import get_imagenet100_loaders
from ..network.model_factory import create_model
from .trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_sweep_pairs(value: str) -> list[tuple[float, float]]:
    """Parse lr:weight_decay pairs from a comma-separated CLI string."""
    pairs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid sweep pair '{item}'. Use format lr:weight_decay, "
                "for example 0.01:0.0001."
            )
        lr_text, wd_text = item.split(":", maxsplit=1)
        pairs.append((float(lr_text), float(wd_text)))
    if not pairs:
        raise ValueError("No valid sweep pairs were provided.")
    return pairs


def parse_finetune_sweep(value: str) -> list[tuple[str, float, float, float]]:
    """Parse name:backbone_lr:head_lr:weight_decay sweep entries."""
    runs = []
    for item in value.split(","):
        fields = [field.strip() for field in item.split(":")]
        if len(fields) != 4 or not all(fields):
            raise ValueError(
                f"Invalid fine-tune sweep entry '{item}'. Use "
                "name:backbone_lr:head_lr:weight_decay."
            )
        name, backbone_lr, head_lr, weight_decay = fields
        runs.append((name, float(backbone_lr), float(head_lr), float(weight_decay)))
    if not runs:
        raise ValueError("No valid fine-tune sweep entries were provided.")
    return runs


def format_float_for_name(value: float) -> str:
    """Make a float safe for experiment directory names."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def run_training(config: dict, logger) -> None:
    """Build data/model/trainer and run one training job."""
    set_seed(config.get("seed", 42))
    logger.info(f"Config: {config}")

    logger.info("Loading ImageNet-100 subset...")
    train_loader, val_loader = get_imagenet100_loaders(config)
    config["selected_classes"] = list(train_loader.dataset.classes)
    config["class_to_idx"] = dict(train_loader.dataset.class_to_idx)
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    logger.info(f"Creating model: {config['model_type']} ({config.get('model_name', '')})")
    model = create_model(config)
    logger.info("Model created successfully.")

    if config.get("init_checkpoint"):
        checkpoint_path = config["init_checkpoint"]
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_classes = checkpoint.get("selected_classes")
        if (
            checkpoint_classes is not None
            and checkpoint_classes != config.get("selected_classes")
        ):
            raise ValueError(
                f"Class list in {checkpoint_path} does not match this run."
            )
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "backbone_state_dict" in checkpoint and hasattr(model, "backbone"):
            model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        else:
            raise KeyError(
                f"{checkpoint_path} contains neither model_state_dict nor "
                "backbone_state_dict."
            )
        logger.info(
            f"Initialized model weights from {checkpoint_path} "
            f"(checkpoint epoch {checkpoint.get('epoch', '?')})"
        )

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()


def main() -> None:
    # Parse args
    parser = argparse.ArgumentParser(description="Train model on ImageNet-100 subset")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    # Allow overriding any config key from CLI
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable early stopping on validation accuracy",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=None,
        help="Minimum validation-accuracy improvement in percentage points",
    )
    parser.add_argument(
        "--sweep_pairs",
        type=str,
        default=None,
        help="Comma-separated lr:weight_decay pairs, e.g. 0.03:0,0.01:0.0001",
    )
    parser.add_argument(
        "--finetune_sweep",
        type=str,
        default=None,
        help=(
            "Comma-separated name:backbone_lr:head_lr:weight_decay entries, "
            "e.g. A:1e-5:1e-4:0.05,B:3e-5:3e-4:0.05"
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Load model weights only and start a new training run",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    for key in [
        "data_root",
        "batch_size",
        "epochs",
        "lr",
        "backbone_lr",
        "head_lr",
        "weight_decay",
        "optimizer",
        "experiment_name",
        "early_stopping",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "device",
        "seed",
        "init_checkpoint",
    ]:
        val = getattr(args, key)
        if val is not None:
            config[key] = val
    if args.resume is not None:
        config["resume_checkpoint"] = args.resume

    # Setup
    logger = setup_logger("main")

    if args.sweep_pairs is not None and args.finetune_sweep is not None:
        parser.error("--sweep_pairs and --finetune_sweep cannot be used together.")

    if args.sweep_pairs is None and args.finetune_sweep is None:
        run_training(config, logger)
        return

    base_experiment_name = config.get("experiment_name", "experiment")
    sweep_output_dir = os.path.join(
        config.get("output_dir", "outputs"),
        base_experiment_name,
    )
    if args.finetune_sweep is not None:
        runs = parse_finetune_sweep(args.finetune_sweep)
        logger.info(
            f"Running fine-tune sweep with {len(runs)} configurations in "
            f"{sweep_output_dir}"
        )
        for run_idx, (name, backbone_lr, head_lr, weight_decay) in enumerate(
            runs,
            start=1,
        ):
            run_config = copy.deepcopy(config)
            run_config["lr"] = backbone_lr
            run_config["backbone_lr"] = backbone_lr
            run_config["head_lr"] = head_lr
            run_config["weight_decay"] = weight_decay
            run_config["resume_checkpoint"] = None
            run_config["output_dir"] = sweep_output_dir
            run_config["experiment_name"] = (
                f"{name}_blr{format_float_for_name(backbone_lr)}"
                f"_hlr{format_float_for_name(head_lr)}"
                f"_wd{format_float_for_name(weight_decay)}"
            )
            logger.info(
                f"Starting fine-tune run {run_idx}/{len(runs)} ({name}): "
                f"backbone_lr={backbone_lr:g}, head_lr={head_lr:g}, "
                f"weight_decay={weight_decay:g}"
            )
            run_training(run_config, logger)
        return

    pairs = parse_sweep_pairs(args.sweep_pairs)
    logger.info(
        f"Running sweep with {len(pairs)} lr/weight_decay pairs in "
        f"{sweep_output_dir}"
    )

    for run_idx, (lr, weight_decay) in enumerate(pairs, start=1):
        run_config = copy.deepcopy(config)
        run_config["lr"] = lr
        run_config["weight_decay"] = weight_decay
        run_config["resume_checkpoint"] = None
        run_config["output_dir"] = sweep_output_dir
        run_config["experiment_name"] = (
            f"lr{format_float_for_name(lr)}"
            f"_wd{format_float_for_name(weight_decay)}"
        )
        logger.info(
            f"Starting sweep run {run_idx}/{len(pairs)}: "
            f"lr={lr:g}, weight_decay={weight_decay:g}, "
            f"scheduler={run_config.get('scheduler', 'none')}"
        )
        run_training(run_config, logger)


if __name__ == "__main__":
    main()
