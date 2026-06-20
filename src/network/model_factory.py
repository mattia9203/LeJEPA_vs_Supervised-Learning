"""Model factory: single entry point for creating models from config."""

import torch.nn as nn

from ..globals import MODEL_TYPE_SUPERVISED, MODEL_TYPE_LEJEPA
from .supervised_vit import SupervisedViT
from .lejepa_vit import LeJEPAViT


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
        )
    elif model_type == MODEL_TYPE_LEJEPA:
        model = LeJEPAViT(
            model_name=model_name or "OK-AI/lejepa-vits16-pretrain-in1k",
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Supported: '{MODEL_TYPE_SUPERVISED}', '{MODEL_TYPE_LEJEPA}'"
        )

    return model
