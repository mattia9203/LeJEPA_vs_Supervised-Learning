"""CLI entry point for ResNet-50 LeJEPA pretraining."""

import argparse
import json

import torch
import yaml

from ..data.imagenet100 import (
    get_imagenet100_multicrop_loader,
    get_imagenet100_probe_loaders,
)
from ..network.lejepa_cnn import LeJEPACNN
from ..utils.seed import set_seed
from .lejepa_trainer import LeJEPATrainer


def load_config(path: str) -> dict:
    with open(path) as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet-50 with LeJEPA")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--micro_batch_size", type=int)
    parser.add_argument("--effective_batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--probe_interval", type=int)
    parser.add_argument("--probe_epochs", type=int)
    parser.add_argument("--sigreg_weight", type=float)
    parser.add_argument("--sigreg_num_slices", type=int)
    parser.add_argument("--experiment_name")
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--resume")
    args = parser.parse_args()

    config = load_config(args.config)
    for key in (
        "epochs",
        "micro_batch_size",
        "effective_batch_size",
        "lr",
        "weight_decay",
        "probe_interval",
        "probe_epochs",
        "sigreg_weight",
        "sigreg_num_slices",
        "experiment_name",
        "init_checkpoint",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.resume:
        config["resume_checkpoint"] = args.resume
    micro = config.get("micro_batch_size", 32)
    effective = config.get("effective_batch_size", 128)
    if effective % micro:
        raise ValueError("effective_batch_size must be divisible by micro_batch_size.")
    config["gradient_accumulation_steps"] = effective // micro

    set_seed(config.get("seed", 42))
    train_loader, selected_classes, class_to_idx = get_imagenet100_multicrop_loader(
        config
    )
    config["selected_classes"] = selected_classes
    config["class_to_idx"] = class_to_idx
    probe_train, probe_val = get_imagenet100_probe_loaders(
        config,
        train_fraction_per_class=config.get("probe_train_fraction_per_class", 0.2),
    )
    if hasattr(probe_train.dataset, "indices"):
        config["probe_train_indices"] = list(probe_train.dataset.indices)
    model = LeJEPACNN(
        proj_hidden_dim=config.get("proj_hidden_dim", 2048),
        proj_output_dim=config.get("proj_output_dim", 256),
    )
    initialization = torch.load(
        config["init_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    if initialization.get("selected_classes") != selected_classes:
        raise ValueError("Shared initialization class list does not match this run.")
    model.backbone.load_state_dict(initialization["backbone_state_dict"])
    config["shared_init_seed"] = initialization["seed"]

    trainer = LeJEPATrainer(
        model,
        train_loader,
        probe_train,
        probe_val,
        config,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
