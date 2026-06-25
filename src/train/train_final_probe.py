"""Train the final classifier on a frozen LeJEPA ResNet-50 backbone."""

import argparse

import torch
import yaml

from ..data.imagenet100 import get_imagenet100_loaders
from ..network.supervised_cnn import SupervisedCNN
from ..utils.seed import set_seed
from .trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone_checkpoint", required=True)
    parser.add_argument("--experiment_name")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    set_seed(config.get("seed", 42))
    train_loader, val_loader = get_imagenet100_loaders(config)
    config["selected_classes"] = list(train_loader.dataset.classes)
    config["class_to_idx"] = dict(train_loader.dataset.class_to_idx)

    model = SupervisedCNN(
        num_classes=config.get("num_classes", 30),
        freeze_backbone=True,
        head_dropout=config.get("head_dropout", 0.0),
    )
    checkpoint = torch.load(
        args.backbone_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
    config["source_backbone_checkpoint"] = args.backbone_checkpoint
    Trainer(model, train_loader, val_loader, config).fit()


if __name__ == "__main__":
    main()
