"""Model factory: single entry point for creating models from config."""

import torch.nn as nn

from ..globals import (
    MODEL_TYPE_LEJEPA,
    MODEL_TYPE_LEJEPA_CNN,
    MODEL_TYPE_SUPERVISED,
    MODEL_TYPE_SUPERVISED_CNN,
)
from .lejepa_cnn import LeJEPACNN
from .supervised_vit import SupervisedViT
from .lejepa_vit import LeJEPAViT
from .supervised_cnn import SupervisedCNN


def create_model(config: dict) -> nn.Module:
    """Instantiate a model based on config['model_type'].

    Supported model types:
        - 'supervised_vit' : timm-based supervised ViT-S/16
        - 'lejepa_vit'     : Hugging Face LeJEPA ViT-S/16

    Returns:
        nn.Module with a .forward(x) → logits interface.
    """
    model_type = config["model_type"]
    model_name = config.get("model_name", None)
    num_classes = config.get("num_classes", 30)
    freeze_backbone = config.get("freeze_backbone", False)

    if model_type == MODEL_TYPE_SUPERVISED:
        model = SupervisedViT(
            model_name=model_name or "vit_small_patch16_224",
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            trainable_last_blocks=config.get("trainable_last_blocks"),
            head_dropout=config.get("head_dropout", 0.0),
        )
    elif model_type == MODEL_TYPE_LEJEPA:
        model = LeJEPAViT(
            model_name=model_name or "OK-AI/lejepa-vits16-pretrain-in1k",
            model_revision=config.get(
                "model_revision",
                "cc7022877d51494709ef398d437fb8619349e0f9",
            ),
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            trainable_last_blocks=config.get("trainable_last_blocks"),
            head_dropout=config.get("head_dropout", 0.0),
        )
    elif model_type == MODEL_TYPE_SUPERVISED_CNN:
        model = SupervisedCNN(
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            head_dropout=config.get("head_dropout", 0.0),
        )
    elif model_type == MODEL_TYPE_LEJEPA_CNN:
        model = LeJEPACNN(
            proj_hidden_dim=config.get("proj_hidden_dim", 2048),
            proj_output_dim=config.get("proj_output_dim", 256),
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Supported: '{MODEL_TYPE_SUPERVISED}', '{MODEL_TYPE_LEJEPA}', "
            f"'{MODEL_TYPE_SUPERVISED_CNN}', '{MODEL_TYPE_LEJEPA_CNN}'"
        )

    return model
